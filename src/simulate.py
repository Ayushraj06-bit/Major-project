"""Scenario simulation: what-if analysis that stays internally coherent.

**The mechanism is not negotiable.** A scenario is applied to the *raw panel*, and
the prediction is then produced by the same path a real forecast takes: rebuild
every derived feature, apply the frozen scaler, predict. This module contains no
feature engineering of its own, and must not grow any.

The reason is specific. If a user raises rainfall by 20% and only ``rainfall_t``
moves, then ``rainfall_lag_1``, ``rainfall_lag_2``, the rolling means and the
neighbouring-state terms all still describe the *old* world. The model is handed a
row that is internally contradictory — a wet month preceded by dry months that its
own lag columns say were wet — a combination that appears nowhere in training. It
will return a number, and that number means nothing. Routing through
:func:`~src.features.build_features` is what makes the whole feature vector move
together.

**Guardrails.** Modified values are clamped to what the data has actually seen,
per state, and the result carries an ``out_of_distribution`` flag when clamping
bit. Beyond the training distribution is exactly where neural networks produce
confident nonsense, and a scenario tool that silently answers "+400% rainfall" is
worse than one that refuses.

This module holds **two different things**, and confusing them would be a serious
error of interpretation:

* :func:`simulate` answers *"what if conditions were different?"* — the user
  supplies a deviation and the model responds to a counterfactual world.
* :func:`forecast_horizon` answers *"what happens next under typical
  conditions?"* — nobody supplies anything. It is the model's forward view, with
  climatology standing in for weather nobody has observed yet.

A scenario is a hypothetical; a forward projection is a forecast. Reporting one as
the other would either invent a policy claim from a plain forecast, or dress a
counterfactual up as a prediction.

.. warning::

   **The model learns correlation, not causation.** Every result here describes
   how *this model* responds to an input it was never shown, not how dengue
   transmission responds to rainfall. "Predicted risk rises when rainfall rises"
   is a statement about learned behaviour. It is not evidence that rainfall causes
   cases, it does not license a claim about intervening on rainfall, and the
   report should say so rather than leave a reader to assume otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from src.config import Config
from src.sources import PANEL_KEYS

#: Scenario change modes.
ABSOLUTE = "absolute"
PERCENT = "percent"
VALID_MODES = frozenset({ABSOLUTE, PERCENT})


class SimulationError(RuntimeError):
    """Raised when a scenario cannot be applied or predicted."""


@dataclass(frozen=True)
class Scenario:
    """One what-if question, expressed against a raw panel variable.

    Attributes:
        variable: Raw panel column to change, e.g. ``"rainfall"``. Not a derived
            feature name — the whole point is that derived columns follow.
        change: Size of the change. ``20`` with ``mode="percent"`` is +20%;
            ``2.0`` with ``mode="absolute"`` adds 2 units.
        mode: ``"percent"`` or ``"absolute"``.
        start: First period affected. ``None`` means from the beginning.
        end: Last period affected. ``None`` means to the end.
        states: States affected. ``None`` means all of them.
        label: Optional human description, carried into the result.
    """

    variable: str
    change: float
    mode: str = PERCENT
    start: date | None = None
    end: date | None = None
    states: tuple[str, ...] | None = None
    label: str = ""

    def __post_init__(self) -> None:
        if self.mode not in VALID_MODES:
            raise SimulationError(
                f"Scenario.mode: expected one of {sorted(VALID_MODES)}, got {self.mode!r}"
            )
        if self.start is not None and self.end is not None and self.start > self.end:
            raise SimulationError(
                f"Scenario: start ({self.start}) is after end ({self.end})"
            )

    @property
    def is_null(self) -> bool:
        """Whether this scenario changes nothing.

        A null scenario must reproduce the baseline forecast exactly; see
        :func:`simulate`.
        """
        return self.change == 0

    def describe(self) -> str:
        """One-line description, for logs and the dashboard."""
        if self.label:
            return self.label
        sign = "+" if self.change >= 0 else ""
        unit = "%" if self.mode == PERCENT else ""
        where = "all states" if self.states is None else ", ".join(self.states)
        when = ""
        if self.start or self.end:
            when = f" from {self.start or 'start'} to {self.end or 'end'}"
        return f"{sign}{self.change:g}{unit} {self.variable} in {where}{when}"


@dataclass(frozen=True)
class SimulationResult:
    """Baseline and scenario forecasts side by side, with the guardrail flags.

    Attributes:
        scenario: The scenario that was applied.
        baseline: Forecasts from the unmodified panel.
        scenario_forecast: Forecasts from the modified panel.
        delta: Per-row change, on both the log and case-rate scales.
        out_of_distribution: True when clamping bit anywhere. Read it with
            :attr:`clamped_fraction`: one clamped cell out of a hundred is a
            different situation from every cell clamped.
        clamped_rows: How many panel cells were clamped.
        modified_rows: How many panel cells the scenario touched.
        affects_model: False when the changed variable feeds no model input, in
            which case a zero delta means "the model does not use this", not "this
            has no effect".

    .. warning::

       Results describe model behaviour under a counterfactual input, not
       epidemiological fact. See the module docstring.
    """

    scenario: Scenario
    baseline: pd.DataFrame
    scenario_forecast: pd.DataFrame
    delta: pd.DataFrame
    out_of_distribution: bool
    clamped_rows: int
    modified_rows: int
    affects_model: bool

    @property
    def clamped_fraction(self) -> float:
        """Share of touched cells that hit the guardrail.

        Reported beside the boolean because the boolean alone cries wolf. Raising
        a record cloudburst by 5% genuinely does leave the observed range, so a
        modest scenario over a spiky variable will clamp a cell or two. The
        distinction that matters to a reader is 2% of cells against 100% of them.
        """
        return 0.0 if self.modified_rows == 0 else self.clamped_rows / self.modified_rows

    @property
    def mean_delta(self) -> float:
        """Mean change in predicted cases per 100,000."""
        return float(self.delta["delta_cases_per_100k"].mean())

    def summary(self) -> str:
        """A readable block, including the warnings that must travel with it."""
        lines = [
            f"Scenario        : {self.scenario.describe()}",
            f"Cells modified  : {self.modified_rows}",
            f"Mean delta      : {self.mean_delta:+.4f} cases per 100,000",
            f"Max increase    : {self.delta['delta_cases_per_100k'].max():+.4f}",
            f"Max decrease    : {self.delta['delta_cases_per_100k'].min():+.4f}",
        ]
        if not self.affects_model:
            lines.append(
                f"NOTE            : {self.scenario.variable!r} feeds no model input, "
                "so a zero delta means the model ignores it, not that it does not matter"
            )
        if self.out_of_distribution:
            lines.append(
                f"OUT OF RANGE    : {self.clamped_rows} of {self.modified_rows} cell(s) "
                f"clamped ({self.clamped_fraction:.0%}) to the observed range. Those "
                "cells lie outside anything the model was trained on, where its "
                "output is not meaningful."
            )
        lines.append(
            "CAVEAT          : the model learns correlation, not causation. This "
            "describes model behaviour, not dengue transmission."
        )
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# The one entry point
# --------------------------------------------------------------------------- #


def simulate(
    panel: pd.DataFrame,
    scenario: Scenario,
    model: Any,
    cfg: Config,
) -> SimulationResult:
    """Apply a scenario to the raw panel and forecast from it.

    **Not** :func:`forecast_horizon`. This answers "what if conditions were
    different?" against a deviation the user chose. That one answers "what happens
    next under typical conditions?" and takes no deviation at all.

    Args:
        panel: Cleaned wide panel, unmodified. Not mutated.
        scenario: The what-if to apply.
        model: A loaded production model exposing ``predict(panel)``. Loaded, never
            constructed here — the explanation and the forecast must come from the
            same fitted weights.
        cfg: Loaded configuration; ``simulate`` supplies the guardrails.

    Returns:
        Baseline and scenario forecasts, their difference, and the guardrail flags.

    Raises:
        SimulationError: the variable is not in the panel, or the scenario selects
            no rows at all.
    """
    if scenario.variable not in panel.columns:
        raise SimulationError(
            f"scenario variable {scenario.variable!r} is not a panel column; "
            f"available: {sorted(panel.columns)}"
        )

    mask = _selection_mask(panel, scenario)
    if not mask.any():
        raise SimulationError(
            f"scenario selects no rows: variable={scenario.variable!r}, "
            f"states={scenario.states}, {scenario.start} to {scenario.end}"
        )

    modified, clamped = apply_scenario(panel, scenario, cfg, mask)

    baseline = model.predict(panel)
    scenario_forecast = model.predict(modified)

    return SimulationResult(
        scenario=scenario,
        baseline=baseline,
        scenario_forecast=scenario_forecast,
        delta=_difference(baseline, scenario_forecast),
        out_of_distribution=bool(clamped > 0),
        clamped_rows=int(clamped),
        modified_rows=int(mask.sum()),
        affects_model=bool(model.spec.columns_from(scenario.variable)),
    )


def apply_scenario(
    panel: pd.DataFrame,
    scenario: Scenario,
    cfg: Config,
    mask: pd.Series | None = None,
) -> tuple[pd.DataFrame, int]:
    """Return a copy of the panel with the scenario applied, and the clamp count.

    Separated from :func:`simulate` so the modification can be inspected and tested
    without running a model.

    Returns:
        ``(modified_panel, n_clamped)``.
    """
    mask = _selection_mask(panel, scenario) if mask is None else mask
    modified = panel.copy()

    original = modified.loc[mask, scenario.variable]
    if scenario.mode == PERCENT:
        proposed = original * (1.0 + scenario.change / 100.0)
    else:
        proposed = original + scenario.change

    clamped_values, n_clamped = _clamp(panel, scenario.variable, proposed, original, cfg)
    modified.loc[mask, scenario.variable] = clamped_values
    return modified, n_clamped


# --------------------------------------------------------------------------- #
# Guardrails
# --------------------------------------------------------------------------- #


def plausible_range(
    panel: pd.DataFrame, variable: str, cfg: Config
) -> pd.DataFrame:
    """Per-state bounds on what a variable has plausibly been.

    Per state, not pooled: Rajasthan and Kerala have entirely different rainfall
    distributions, and a bound wide enough for both is no bound at all.

    ``minmax`` uses the observed extremes; ``sigma`` uses mean +/- ``clamp_n_sigma``
    standard deviations, which is tighter and refuses more.
    """
    grouped = panel.groupby(level="state")[variable]
    if cfg.simulate.clamp_strategy == "minmax":
        return pd.DataFrame({"low": grouped.min(), "high": grouped.max()})

    mean, spread = grouped.mean(), grouped.std()
    return pd.DataFrame(
        {
            "low": mean - cfg.simulate.clamp_n_sigma * spread,
            "high": mean + cfg.simulate.clamp_n_sigma * spread,
        }
    )


def _clamp(
    panel: pd.DataFrame,
    variable: str,
    proposed: pd.Series,
    original: pd.Series,
    cfg: Config,
) -> tuple[pd.Series, int]:
    """Hold proposed values inside the observed range, and count what was held.

    The bound is widened to include each cell's own original value. Without that,
    a sigma-based clamp would pull genuine historical extremes back toward the
    mean, and a scenario that changes *nothing* would still alter the panel — so a
    zero-change scenario would not reproduce the baseline forecast. The clamp
    exists to stop a user inventing conditions, not to overwrite what happened.
    """
    bounds = plausible_range(panel, variable, cfg)
    states = proposed.index.get_level_values("state")

    low = np.minimum(bounds.loc[states, "low"].to_numpy(), original.to_numpy())
    high = np.maximum(bounds.loc[states, "high"].to_numpy(), original.to_numpy())

    values = proposed.to_numpy()
    clamped = np.clip(values, low, high)
    n_clamped = int(np.count_nonzero(~np.isclose(clamped, values, equal_nan=True)))
    return pd.Series(clamped, index=proposed.index), n_clamped


def _selection_mask(panel: pd.DataFrame, scenario: Scenario) -> pd.Series:
    """Which panel rows the scenario touches."""
    states = panel.index.get_level_values("state")
    dates = panel.index.get_level_values("date")

    mask = pd.Series(True, index=panel.index)
    if scenario.states is not None:
        mask &= states.isin(scenario.states)
    if scenario.start is not None:
        mask &= dates >= pd.Timestamp(scenario.start)
    if scenario.end is not None:
        mask &= dates <= pd.Timestamp(scenario.end)
    return mask


def _difference(baseline: pd.DataFrame, scenario: pd.DataFrame) -> pd.DataFrame:
    """Per-row change between two forecast frames, aligned on state and origin."""
    keys = ["state", "origin_date"]
    merged = baseline.merge(scenario, on=keys, suffixes=("_base", "_scenario"))

    delta = merged[keys].copy()
    delta["delta_log"] = merged["predicted_log_scenario"] - merged["predicted_log_base"]
    if "predicted_cases_per_100k_base" in merged:
        delta["delta_cases_per_100k"] = (
            merged["predicted_cases_per_100k_scenario"]
            - merged["predicted_cases_per_100k_base"]
        )
    else:
        delta["delta_cases_per_100k"] = delta["delta_log"]

    for side in ("base", "scenario"):
        delta[f"predicted_{side}"] = merged[f"predicted_log_{side}"]
        delta[f"lower_{side}"] = merged[f"lower_log_{side}"]
        delta[f"upper_{side}"] = merged[f"upper_log_{side}"]
    return delta


# --------------------------------------------------------------------------- #
# Forward projection — deliberately not scenario simulation
# --------------------------------------------------------------------------- #

#: Labels distinguishing how far a step had to reach for its inputs.
DIRECT = "direct"
RECURSIVE = "recursive"


@dataclass(frozen=True)
class ForecastStep:
    """One period of a forward projection.

    Attributes:
        target_date: The period being predicted.
        steps_ahead: How many periods past the last observation this is.
        mode: ``direct`` when every input was observed, ``recursive`` when the
            model was fed its own earlier prediction. The distinction must survive
            to the screen: a five-month recursive projection and a one-month direct
            forecast are not the same kind of claim.
        reliability: 1.0 for a direct step, decaying with recursive depth.
    """

    target_date: pd.Timestamp
    steps_ahead: int
    predicted_cases_per_100k: float
    lower_cases_per_100k: float
    upper_cases_per_100k: float
    mode: str
    reliability: float

    @property
    def is_recursive(self) -> bool:
        """Whether this step was fed the model's own output."""
        return self.mode == RECURSIVE


@dataclass(frozen=True)
class ForecastCurve:
    """A state's forward path, every step from the last observation onward.

    The whole curve rather than a single endpoint: the shape between here and
    there is the useful part, and a lone number four months out invites being read
    with a confidence the method does not support.

    Attributes:
        last_observed: Final period with a real observation.
        direct_until: Last period reachable without feeding predictions back.
        truncated: True when the request was capped at
            ``forecast.max_recursive_steps``.
    """

    state: str
    steps: tuple[ForecastStep, ...]
    last_observed: pd.Timestamp
    direct_until: pd.Timestamp
    truncated: bool

    @property
    def reliability(self) -> float:
        """Reliability of the furthest step, which is the weakest link."""
        return min((step.reliability for step in self.steps), default=0.0)

    @property
    def has_recursive(self) -> bool:
        """Whether any step was produced recursively."""
        return any(step.is_recursive for step in self.steps)

    def frame(self) -> pd.DataFrame:
        """The curve as a table, for charting and export."""
        return pd.DataFrame(
            [
                {
                    "target_date": step.target_date,
                    "steps_ahead": step.steps_ahead,
                    "predicted": step.predicted_cases_per_100k,
                    "lower": step.lower_cases_per_100k,
                    "upper": step.upper_cases_per_100k,
                    "mode": step.mode,
                    "reliability": step.reliability,
                }
                for step in self.steps
            ]
        )

    def at(self, target_date: pd.Timestamp) -> ForecastStep | None:
        """The step for one period, or None if the curve does not reach it."""
        for step in self.steps:
            if step.target_date == pd.Timestamp(target_date):
                return step
        return None

    def describe(self) -> str:
        """A readable summary carrying the caveats that must travel with it."""
        lines = [
            f"State           : {self.state}",
            f"Last observed   : {self.last_observed.date()}",
            f"Direct until    : {self.direct_until.date()}",
            f"Steps           : {len(self.steps)}",
        ]
        if self.has_recursive:
            recursive = sum(1 for step in self.steps if step.is_recursive)
            lines.append(
                f"RECURSIVE       : {recursive} step(s) fed the model its own output "
                f"with climatological weather. Reliability {self.reliability:.0%}."
            )
        if self.truncated:
            lines.append("TRUNCATED       : the request went past the configured cap.")
        lines.append(
            "CAVEAT          : a forward projection under typical conditions, not a "
            "scenario. Recursive steps compound their own error."
        )
        return "\n".join(lines)


def forecast_horizon(
    panel: pd.DataFrame,
    state: str,
    target_date: pd.Timestamp | date,
    model: Any,
    cfg: Config,
) -> ForecastCurve:
    """Project one state forward to a target date, under typical conditions.

    **Not** :func:`simulate`. Nothing here is a counterfactual: no deviation is
    applied, and the climate inputs past the last observation are the state's own
    historical normals for that calendar period. This is the model's forward view,
    not an answer to a what-if.

    Two regimes, and the difference matters:

    * **Direct**, up to ``last observation + the trained horizon``. Every input was
      really observed, and the conformal interval carries its usual meaning.
    * **Recursive**, past that. Each step's prediction is fed back as the next
      step's case history, and climatological normals stand in for weather nobody
      has measured. Error compounds, so the interval is widened at every step and
      the reliability flag falls.

    The interval widening uses a random-walk approximation, scaling the calibrated
    half-width by the square root of the step count. **This is not a conformal
    guarantee.** Split conformal is valid for the horizon it was calibrated on;
    nothing about it extends to a model consuming its own output. The widening is
    an honest acknowledgement that the interval must grow, not a claim about
    coverage, and the report should say so.

    Args:
        panel: Cleaned wide panel. Not mutated.
        state: Which state to project.
        target_date: How far to project. Capped at
            ``forecast.max_recursive_steps`` past the direct horizon.
        model: A loaded production model exposing ``predict(panel)`` and ``spec``.
        cfg: Loaded configuration.

    Returns:
        The full :class:`ForecastCurve`, every step from the first forecastable
        period to the target.

    Raises:
        SimulationError: the state is absent, or it has no observed history to
            project from.
    """
    states = set(panel.index.get_level_values("state"))
    if state not in states:
        raise SimulationError(
            f"{state!r} is not in the panel; available: {sorted(states)}"
        )

    last_observed = _last_observed(panel, state, cfg)
    horizon = int(model.spec.horizon)
    offset = _period_offset(cfg)
    normals = climatological_normals(panel, cfg)

    direct_until = last_observed + offset * horizon
    wanted = pd.Timestamp(target_date)
    cap = direct_until + offset * cfg.forecast.max_recursive_steps
    truncated = wanted > cap
    reach = min(wanted, cap)

    # The panel has to be extended before anything can be predicted at all.
    # build_features drops a window whose target is missing, so with a panel that
    # stops at the last observation the furthest origin it will emit is one whose
    # answer is already known. Appending `horizon` periods of climatology creates
    # the target slots; the windows feeding them are still made entirely of real
    # observations, which is what keeps these steps direct.
    working = panel
    for step in range(1, horizon + 1):
        working = _extend_panel(working, last_observed + offset * step, normals, cfg)

    prediction = model.predict(working)
    steps = _direct_steps(prediction, state, last_observed, reach, offset)

    if reach > direct_until:
        steps.extend(
            _recursive_steps(
                working, prediction, state, last_observed, direct_until, reach,
                model, cfg, offset, normals,
            )
        )

    return ForecastCurve(
        state=state,
        steps=tuple(steps),
        last_observed=last_observed,
        direct_until=direct_until,
        truncated=truncated,
    )


def _direct_steps(
    baseline: pd.DataFrame,
    state: str,
    last_observed: pd.Timestamp,
    reach: pd.Timestamp,
    offset: pd.DateOffset,
) -> list[ForecastStep]:
    """Steps the model reaches from observed inputs alone.

    Kept to origins at or before the last observation, so every value in the
    window was really measured.
    """
    rows = baseline[baseline["state"] == state].copy()
    rows["target_date"] = pd.to_datetime(rows["target_date"])
    rows["origin_date"] = pd.to_datetime(rows["origin_date"])
    rows = rows[
        (rows["target_date"] > last_observed)
        & (rows["target_date"] <= reach)
        & (rows["origin_date"] <= last_observed)
    ].sort_values("target_date")

    return [
        ForecastStep(
            target_date=row.target_date,
            steps_ahead=_periods_between(last_observed, row.target_date, offset),
            predicted_cases_per_100k=float(row.predicted_cases_per_100k),
            lower_cases_per_100k=float(row.lower_cases_per_100k),
            upper_cases_per_100k=float(row.upper_cases_per_100k),
            mode=DIRECT,
            reliability=1.0,
        )
        for row in rows.itertuples()
    ]


def _recursive_steps(
    working: pd.DataFrame,
    prediction: pd.DataFrame,
    state: str,
    last_observed: pd.Timestamp,
    direct_until: pd.Timestamp,
    reach: pd.Timestamp,
    model: Any,
    cfg: Config,
    offset: pd.DateOffset,
    normals: pd.DataFrame,
) -> list[ForecastStep]:
    """Steps that feed the model its own output, one period at a time.

    Each iteration writes the previous prediction back into the panel as the case
    count for that period, then extends by one more period of climatology and
    re-runs :func:`~src.features.build_features` through ``model.predict``. Nothing
    here reimplements a lag: a fed-back case value has to move every derived column
    that depends on cases, and going back through the builder is the only way to
    guarantee it does.

    The feed-back is applied for **every** state, not only the one being projected.
    The model is pooled and carries spatial terms, so leaving neighbours frozen at
    climatology while one state advances would quietly change what the spatial lags
    are reporting.

    Each iteration runs the model **once**. The value written back for a period is
    the one the previous iteration already predicted for it, so it is carried in
    rather than recomputed -- re-predicting to obtain a number already in hand
    would rebuild the entire feature pipeline for nothing.

    Args:
        working: Panel extended through ``direct_until``.
        prediction: The predictions already made from ``working``; supplies the
            first feed-back value.
        state: The state being projected.
        last_observed: Final period with a real observation, the origin every
            step counts from.
        direct_until: Last period reachable without feeding predictions back.
        reach: Furthest period to project to.
        model: Loaded production model.
        cfg: Loaded configuration.
        offset: One period at the configured granularity.
        normals: Per-state per-calendar-period climatological means.
    """
    steps: list[ForecastStep] = []
    period = direct_until
    depth = 0

    while period < reach:
        working = _feed_back(working, prediction, period, cfg)
        period = period + offset
        depth += 1

        working = _extend_panel(working, period, normals, cfg)
        prediction = model.predict(working)

        rows = prediction[prediction["state"] == state].copy()
        rows["target_date"] = pd.to_datetime(rows["target_date"])
        row = rows[rows["target_date"] == period]
        if row.empty:
            break

        point = float(row["predicted_cases_per_100k"].iloc[0])
        if not np.isfinite(point):
            break
        half = _widened_half_width(row, depth)
        steps.append(
            ForecastStep(
                target_date=period,
                steps_ahead=_periods_between(last_observed, period, offset),
                predicted_cases_per_100k=point,
                lower_cases_per_100k=max(point - half, 0.0),
                upper_cases_per_100k=point + half,
                mode=RECURSIVE,
                reliability=_reliability(depth, cfg),
            )
        )

    return steps


def _feed_back(
    working: pd.DataFrame,
    prediction: pd.DataFrame,
    period: pd.Timestamp,
    cfg: Config,
) -> pd.DataFrame:
    """Write the model's prediction for ``period`` back as that period's cases.

    This is what makes the projection recursive. The prediction is passed in
    rather than recomputed: the caller already holds it, and running the model
    again to read a number it just produced would rebuild every feature for
    nothing. The prediction is a rate, so it
    is converted back to a count using each state's population before it is stored,
    keeping the panel in the units every other stage expects.

    The fed-back value is held inside :func:`plausible_range`, the same clamp a
    scenario gets. A recursive loop is a feedback loop: a model that reads its own
    output as an input can amplify it every round and diverge to a number no state
    has ever seen. Clamping to the observed range does not make a runaway
    projection correct, but it keeps a divergence visible as a flat line at the
    historical maximum instead of an overflow, and the reliability flag has
    already said how much to trust that far out.
    """
    rows = prediction.copy()
    rows["target_date"] = pd.to_datetime(rows["target_date"])
    rows = rows[rows["target_date"] == period]
    if rows.empty:
        return working

    observed = working[working.index.get_level_values("date") < period]
    bounds = plausible_range(observed, cfg.data.target_column, cfg)

    updated = working.copy()
    for row in rows.itertuples():
        key = (row.state, period)
        if key not in updated.index or key[0] not in bounds.index:
            continue
        rate = float(row.predicted_cases_per_100k)
        if not np.isfinite(rate):
            continue
        population = float(updated.loc[key, "population"])
        count = rate * population / cfg.data.population_normalisation
        updated.loc[key, cfg.data.target_column] = float(
            np.clip(count, bounds.loc[key[0], "low"], bounds.loc[key[0], "high"])
        )
    return updated


def _periods_between(
    start: pd.Timestamp, end: pd.Timestamp, offset: pd.DateOffset
) -> int:
    """How many whole periods separate two dates."""
    count = 0
    cursor = start
    while cursor < end:
        cursor = cursor + offset
        count += 1
    return count


def _widened_half_width(row: pd.DataFrame, depth: int) -> float:
    """Half-width of the interval at a recursive depth.

    Scaled by ``sqrt(depth)``, the random-walk rate at which independent
    step errors accumulate. An approximation, and stated as one: the conformal
    guarantee does not survive a model consuming its own output, so this is here to
    stop a flat band implying constant confidence four months out.
    """
    base = float(row["upper_cases_per_100k"].iloc[0] - row["predicted_cases_per_100k"].iloc[0])
    return base * float(np.sqrt(depth + 1))


def _reliability(depth: int, cfg: Config) -> float:
    """Reliability at a recursive depth, decaying to zero at the cap."""
    cap = max(cfg.forecast.max_recursive_steps, 1)
    return float(max(0.0, 1.0 - depth / (cap + 1)))


def climatological_normals(panel: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Each state's historical mean of every variable, per calendar period.

    What stands in for weather nobody has observed yet. A state's own normals
    rather than a national average, because "a typical October" means something
    different in Kerala and in Punjab.
    """
    frame = panel.copy()
    dates = pd.DatetimeIndex(frame.index.get_level_values("date"))
    frame["_period"] = (
        dates.month if cfg.project.granularity == "monthly" else dates.isocalendar().week
    )
    grouped = frame.groupby([frame.index.get_level_values("state"), "_period"]).mean()
    grouped.index.names = ["state", "_period"]
    return grouped


def _extend_panel(
    panel: pd.DataFrame,
    period: pd.Timestamp,
    normals: pd.DataFrame,
    cfg: Config,
) -> pd.DataFrame:
    """Append one period for every state, filled with climatological normals.

    Every state, not only the one being projected: the panel has to stay
    rectangular for the spatial-lag terms, and a ragged one would quietly change
    which neighbours a state appears to have.
    """
    calendar = (
        period.month
        if cfg.project.granularity == "monthly"
        else int(pd.Timestamp(period).isocalendar().week)
    )
    states = sorted(set(panel.index.get_level_values("state")))

    rows = []
    for state in states:
        key = (state, calendar)
        values = (
            normals.loc[key].to_dict()
            if key in normals.index
            else panel.loc[state].mean().to_dict()
        )
        rows.append({column: values.get(column, np.nan) for column in panel.columns})

    addition = pd.DataFrame(
        rows,
        index=pd.MultiIndex.from_product([states, [period]], names=list(PANEL_KEYS)),
    )
    return pd.concat([panel, addition]).sort_index()


def _last_observed(panel: pd.DataFrame, state: str, cfg: Config) -> pd.Timestamp:
    """Final period where this state has a real case observation.

    Raises:
        SimulationError: the state has no observations to project from.
    """
    series = panel.loc[state, cfg.data.target_column].dropna()
    if series.empty:
        raise SimulationError(
            f"{state!r} has no observed {cfg.data.target_column} to project from"
        )
    return pd.Timestamp(series.index.max())


def _period_offset(cfg: Config) -> pd.DateOffset:
    """One period, at the configured granularity."""
    if cfg.project.granularity == "monthly":
        return pd.DateOffset(months=1)
    return pd.DateOffset(weeks=1)
