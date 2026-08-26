"""Decision support: turning a forecast into a defensible recommended action.

This is the component most likely to be challenged, because it is the one that
tells somebody what to do. The defence is that **every clause traces to a number**:

    HIGH - predicted 88 cases per 100k (80% interval upper bound 140), above the
    90th historical percentile of 62. Top drivers: rainfall lag-3, humidity lag-2.
    Actions: source-reduction drive; hospital bed pre-positioning; targeted
    awareness messaging.

Three design commitments make that possible.

**Thresholds are derived from data, never chosen.** Each state is compared against
its own history — quantiles of its observed case rate, or an EWMA control limit.
A hand-written ``if cases > 50`` cannot answer "why fifty?"; a quantile can answer
"because nine in ten weeks on record were below it, here".

**Tiers are assigned on the interval's upper bound, not the point forecast.** For
preparedness the relevant question is the plausible worst case, not the
expectation. A point forecast that under-predicts spikes — which the sudden-outbreak
analysis shows real dengue models do — would systematically alert late.

**Recommendations are objects, not strings.** :func:`render` is a separate
function. The dashboard needs the tier, the trigger value and the threshold as
data it can sort and colour, not a sentence it has to parse back apart.

The action catalogue lives in ``config.yaml`` under ``risk.actions``, so a domain
expert can revise it without touching code. What this module owns is the mapping
from evidence to tier; what actions a tier warrants is a public-health judgement,
not a modelling one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.config import Config
from src.features import target_level

#: Label for the case below any tier boundary.
BASELINE_LABEL = "below the lowest threshold"


class RecommendationError(RuntimeError):
    """Raised when recommendations cannot be produced from the inputs given."""


@dataclass(frozen=True)
class Threshold:
    """One tier boundary for one state, and the evidence behind it.

    Attributes:
        state: Which state this applies to.
        tier: The tier entered when this boundary is crossed.
        value_log: Boundary on the modelled scale, where comparison happens.
        value_cases_per_100k: The same boundary, back-transformed for reading.
        label: How the boundary was derived, e.g. "90th historical percentile".
        n_observations: How many historical points it rests on. A quantile from
            eight observations is not a quantile, and a reader deserves to know.
    """

    state: str
    tier: str
    value_log: float
    value_cases_per_100k: float
    label: str
    n_observations: int


@dataclass(frozen=True)
class Thresholds:
    """Every tier boundary, keyed by state, plus the method that produced them."""

    method: str
    by_state: dict[str, tuple[Threshold, ...]]

    def for_state(self, state: str) -> tuple[Threshold, ...]:
        """Boundaries for one state, ascending.

        Raises:
            RecommendationError: no history was available for that state.
        """
        try:
            return self.by_state[state]
        except KeyError:
            raise RecommendationError(
                f"no thresholds for {state!r}; it had no historical observations, so "
                "there is nothing to compare a forecast against"
            ) from None

    def frame(self) -> pd.DataFrame:
        """All boundaries as a table, for the report and the dashboard."""
        return pd.DataFrame(
            [
                {
                    "state": threshold.state,
                    "tier": threshold.tier,
                    "value_log": threshold.value_log,
                    "value_cases_per_100k": threshold.value_cases_per_100k,
                    "label": threshold.label,
                    "n_observations": threshold.n_observations,
                }
                for thresholds in self.by_state.values()
                for threshold in thresholds
            ]
        )


@dataclass(frozen=True)
class Recommendation:
    """One state, one forecast period, and what follows from it.

    Machine-readable by design: the dashboard sorts and colours on these fields,
    and :func:`render` is the only thing that turns them into a sentence.

    Attributes:
        trigger_value_log: The number that was compared against the threshold.
            Named explicitly so "which number triggered this?" always has an
            answer.
        trigger_basis: Whether that number was the interval upper bound or the
            point forecast.
        threshold_crossed: The boundary it exceeded, or None at baseline tier.
    """

    state: str
    origin_date: pd.Timestamp
    target_date: pd.Timestamp
    predicted_log: float
    predicted_cases_per_100k: float
    lower_cases_per_100k: float
    upper_cases_per_100k: float
    interval_coverage: float
    tier: str
    trigger_value_log: float
    trigger_value_cases_per_100k: float
    trigger_basis: str
    threshold_crossed: Threshold | None
    drivers: tuple[tuple[str, float], ...]
    actions: tuple[str, ...]
    action_source: str

    @property
    def is_alert(self) -> bool:
        """Whether any threshold was crossed at all."""
        return self.threshold_crossed is not None

    def evidence(self) -> dict[str, Any]:
        """Every number behind this recommendation, flat and quotable.

        The review gate in one method: for any recommendation, this names the
        value that triggered it, what it was compared against, and how that
        comparison value was derived.
        """
        return {
            "state": self.state,
            "target_date": str(self.target_date.date()),
            "tier": self.tier,
            "trigger_value_cases_per_100k": self.trigger_value_cases_per_100k,
            "trigger_basis": self.trigger_basis,
            "threshold_value_cases_per_100k": (
                self.threshold_crossed.value_cases_per_100k if self.threshold_crossed else None
            ),
            "threshold_label": (
                self.threshold_crossed.label if self.threshold_crossed else BASELINE_LABEL
            ),
            "threshold_observations": (
                self.threshold_crossed.n_observations if self.threshold_crossed else None
            ),
            "top_drivers": [label for label, _ in self.drivers],
        }


# --------------------------------------------------------------------------- #
# Thresholds, derived from data
# --------------------------------------------------------------------------- #


def compute_thresholds(panel: pd.DataFrame, cfg: Config) -> Thresholds:
    """Derive per-state tier boundaries from that state's own history.

    Per state because a case rate that is unremarkable in Kerala may be a serious
    outbreak in Punjab. A single national threshold would over-alert one and
    under-alert the other.

    Args:
        panel: Cleaned wide panel carrying cases and population.
        cfg: Loaded configuration; ``risk`` supplies the method.

    Returns:
        The boundaries, with the evidence behind each.

    Raises:
        RecommendationError: the method is unknown, or no state has any history.
    """
    levels = target_level(panel, cfg)
    builders = {"quantile": _quantile_thresholds, "ewma": _ewma_thresholds}
    if cfg.risk.method not in builders:
        raise RecommendationError(
            f"risk.method={cfg.risk.method!r} is not implemented; known methods are "
            f"{sorted(builders)}. The Farrington algorithm is the other established "
            "option and is deliberately not half-implemented here."
        )

    by_state = builders[cfg.risk.method](levels, cfg)
    if not by_state:
        raise RecommendationError("no state had enough history to derive a threshold")
    return Thresholds(method=cfg.risk.method, by_state=by_state)


def _quantile_thresholds(
    levels: pd.Series, cfg: Config
) -> dict[str, tuple[Threshold, ...]]:
    """Boundaries at quantiles of each state's observed case rate.

    Answers "why this number?" with "because that share of weeks on record were
    below it, in this state".
    """
    out: dict[str, tuple[Threshold, ...]] = {}
    for state, series in levels.groupby(level="state"):
        observed = series.dropna()
        if observed.empty:
            continue
        out[state] = tuple(
            Threshold(
                state=str(state),
                tier=tier,
                value_log=float(observed.quantile(quantile)),
                value_cases_per_100k=float(np.expm1(observed.quantile(quantile))),
                label=f"{quantile:.0%} historical percentile",
                n_observations=int(len(observed)),
            )
            # Quantile i is the boundary into tier i+1: the lowest tier is what
            # you are in when no boundary has been crossed.
            for quantile, tier in zip(cfg.risk.quantiles, cfg.risk.tiers[1:], strict=True)
        )
    return out


def _ewma_thresholds(levels: pd.Series, cfg: Config) -> dict[str, tuple[Threshold, ...]]:
    """Control limits at EWMA plus k standard deviations of the residual.

    The classic outbreak-detection form: an observation is unusual relative to
    where the series has recently been, rather than to its whole history. Adapts
    when transmission shifts, at the cost of being slower to flag a slow build-up.
    """
    out: dict[str, tuple[Threshold, ...]] = {}
    for state, series in levels.groupby(level="state"):
        observed = series.dropna()
        if len(observed) < 2:
            continue
        smoothed = observed.ewm(alpha=cfg.risk.ewma_alpha, adjust=False).mean()
        residual_sd = float((observed - smoothed).std(ddof=1))
        centre = float(smoothed.iloc[-1])

        out[state] = tuple(
            Threshold(
                state=str(state),
                tier=tier,
                value_log=centre + multiple * cfg.risk.ewma_sigma * residual_sd,
                value_cases_per_100k=float(
                    np.expm1(centre + multiple * cfg.risk.ewma_sigma * residual_sd)
                ),
                label=f"EWMA + {multiple * cfg.risk.ewma_sigma:g} sigma",
                n_observations=int(len(observed)),
            )
            for multiple, tier in zip(
                range(1, len(cfg.risk.tiers)), cfg.risk.tiers[1:], strict=True
            )
        )
    return out


# --------------------------------------------------------------------------- #
# Recommendations
# --------------------------------------------------------------------------- #


def recommend(
    forecasts: pd.DataFrame,
    thresholds: Thresholds,
    cfg: Config,
    *,
    drivers: dict[tuple[str, pd.Timestamp], tuple[tuple[str, float], ...]] | None = None,
    top_k: int = 2,
) -> list[Recommendation]:
    """Turn forecasts into structured recommendations.

    Args:
        forecasts: Output of ``ProductionModel.predict``, carrying the point
            forecast and interval per ``(state, origin_date)``.
        thresholds: Data-derived boundaries from :func:`compute_thresholds`.
        cfg: Loaded configuration; ``risk`` supplies tiers and actions.
        drivers: Optional SHAP drivers keyed by ``(state, origin_date)``, from
            :mod:`src.explain`. Absent drivers simply omit that clause rather
            than inventing one.
        top_k: How many drivers to quote.

    Returns:
        One :class:`Recommendation` per forecast row.

    Raises:
        RecommendationError: the forecast frame is missing a required column.
    """
    required = {"state", "origin_date", "target_date", "predicted_log"}
    missing = sorted(required - set(forecasts.columns))
    if missing:
        raise RecommendationError(f"forecasts are missing column(s) {missing}")

    basis = cfg.risk.alert_on
    basis_column = "upper_log" if basis == "upper" else "predicted_log"
    if basis_column not in forecasts.columns:
        raise RecommendationError(
            f"risk.alert_on={basis!r} needs column {basis_column!r}, which this "
            "forecast frame does not carry. Wrap the model in ConformalForecaster "
            "so it produces intervals."
        )

    drivers = drivers or {}
    out: list[Recommendation] = []

    for row in forecasts.itertuples():
        trigger = float(getattr(row, basis_column))
        tier, crossed = assign_tier(trigger, thresholds.for_state(row.state), cfg)
        key = (row.state, pd.Timestamp(row.origin_date))

        out.append(
            Recommendation(
                state=str(row.state),
                origin_date=pd.Timestamp(row.origin_date),
                target_date=pd.Timestamp(row.target_date),
                predicted_log=float(row.predicted_log),
                predicted_cases_per_100k=_rate(row, "predicted"),
                lower_cases_per_100k=_rate(row, "lower"),
                upper_cases_per_100k=_rate(row, "upper"),
                interval_coverage=1.0 - cfg.conformal.alpha,
                tier=tier,
                trigger_value_log=trigger,
                trigger_value_cases_per_100k=float(np.expm1(trigger)),
                trigger_basis=(
                    f"{1 - cfg.conformal.alpha:.0%} interval upper bound"
                    if basis == "upper"
                    else "point forecast"
                ),
                threshold_crossed=crossed,
                drivers=tuple(drivers.get(key, ())[:top_k]),
                actions=cfg.risk.actions_for(tier),
                action_source=cfg.risk.action_source,
            )
        )
    return out


def assign_tier(
    value: float, thresholds: tuple[Threshold, ...], cfg: Config
) -> tuple[str, Threshold | None]:
    """Place a value in a tier, returning the boundary it crossed.

    Returns the *highest* boundary crossed, so the recommendation quotes the one
    that actually justifies the tier rather than the first one passed.
    """
    crossed = [threshold for threshold in thresholds if value >= threshold.value_log]
    if not crossed:
        return cfg.risk.tiers[0], None
    highest = max(crossed, key=lambda threshold: threshold.value_log)
    return highest.tier, highest


def _rate(row: Any, prefix: str) -> float:
    """Read a case-rate column, back-transforming from log if needed."""
    direct = getattr(row, f"{prefix}_cases_per_100k", None)
    if direct is not None and np.isfinite(direct):
        return float(direct)
    log_value = getattr(row, f"{prefix}_log", None)
    return float(np.expm1(log_value)) if log_value is not None else float("nan")


# --------------------------------------------------------------------------- #
# Rendering, deliberately separate
# --------------------------------------------------------------------------- #


def render(recommendation: Recommendation) -> str:
    """Render one recommendation as the target sentence.

    Kept apart from the object so the dashboard can style the same facts its own
    way. Every clause here comes from a field; nothing is computed at render time.
    """
    parts = [
        f"{recommendation.tier} - predicted "
        f"{recommendation.predicted_cases_per_100k:.1f} cases per 100k "
        f"({recommendation.interval_coverage:.0%} interval upper bound "
        f"{recommendation.upper_cases_per_100k:.1f})"
    ]

    if recommendation.threshold_crossed is not None:
        threshold = recommendation.threshold_crossed
        parts.append(
            f", above the {threshold.label} of "
            f"{threshold.value_cases_per_100k:.1f} for {recommendation.state} "
            f"(n={threshold.n_observations})"
        )
    else:
        parts.append(
            f", {BASELINE_LABEL} for {recommendation.state}"
        )
    parts.append(".")

    if recommendation.drivers:
        named = ", ".join(label for label, _ in recommendation.drivers)
        parts.append(f" Top drivers: {named}.")

    parts.append(" Actions: " + "; ".join(recommendation.actions) + ".")
    return "".join(parts)


def render_all(recommendations: list[Recommendation]) -> str:
    """Render several recommendations, highest tier first."""
    order = {tier: position for position, tier in enumerate(reversed(_tiers(recommendations)))}
    ranked = sorted(
        recommendations,
        key=lambda item: (order.get(item.tier, 0), item.trigger_value_log),
        reverse=True,
    )
    return "\n".join(render(item) for item in ranked)


def _tiers(recommendations: list[Recommendation]) -> list[str]:
    """Tier names in the order they appear, for ranking."""
    seen: list[str] = []
    for item in recommendations:
        if item.tier not in seen:
            seen.append(item.tier)
    return seen


def alert_summary(recommendations: list[Recommendation], cfg: Config) -> pd.DataFrame:
    """Realised tier rates against the rates the thresholds nominally imply.

    Worth looking at every run, because the two do not match and the gap is
    structural rather than a bug.

    Thresholds are quantiles of the *observed* case rate, so a 90th-percentile
    boundary describes a level exceeded in one period out of ten. But tiers are
    assigned on the interval's **upper bound**, which sits above the point forecast
    by the conformal width. Comparing an upper bound against an observed quantile
    is not comparing like with like, and it alerts materially more often than the
    quantile suggests.

    That is the intended posture — brain.md D-11 chooses the plausible worst case
    on purpose, because a point forecast that under-predicts spikes alerts late —
    but it has a cost in alert fatigue, and the trade is one for a public-health
    reader to make with the numbers in front of them rather than one to bury.

    Returns:
        One row per tier: realised count and share, the share the quantiles imply,
        and the ratio between them.
    """
    if not recommendations:
        raise RecommendationError("no recommendations to summarise")

    counts = pd.Series([item.tier for item in recommendations]).value_counts()
    total = len(recommendations)

    # Quantile q is the boundary into the tier above it, so the nominal share of a
    # tier is the gap between its own boundary and the next one up.
    edges = [0.0, *cfg.risk.quantiles, 1.0]
    nominal = {
        tier: edges[position + 1] - edges[position]
        for position, tier in enumerate(cfg.risk.tiers)
    }

    return pd.DataFrame(
        [
            {
                "tier": tier,
                "count": int(counts.get(tier, 0)),
                "realised_share": counts.get(tier, 0) / total,
                "nominal_share": nominal.get(tier, float("nan")),
                "ratio": (
                    (counts.get(tier, 0) / total) / nominal[tier]
                    if nominal.get(tier)
                    else float("nan")
                ),
            }
            for tier in cfg.risk.tiers
        ]
    )


def to_frame(recommendations: list[Recommendation]) -> pd.DataFrame:
    """Recommendations as a table, one row each, for the dashboard and the report."""
    return pd.DataFrame([item.evidence() for item in recommendations])
