"""A seasonal-profile forecaster, and why it is not the LSTM.

This is a **second, different model**, not the LSTM reaching further. It answers a
different question and it must be labelled differently wherever it appears:

* The LSTM answers *"given the last twelve months of climate, cases and search
  interest in this state, what happens next?"* It is conditional on current
  conditions, and past its trained horizon it can only continue by reading its own
  output, which is why that is capped at a few steps.
* This answers *"what does a typical September look like in this state, given how
  the year is currently running?"* It is a **pattern**, not a conditional forecast.

The consequence is the point. A recursive projection compounds its own error, so
six steps is already a weak claim. A seasonal profile does not compound -- it
repeats -- so it can honestly run two years. What it gives up is responsiveness:
it cannot see an unusual monsoon coming, and it will miss an outbreak that the
calendar did not predict. Both properties have to survive to the screen.

**Level anchor rather than a trend slope.** The obvious design is a linear trend
extrapolated forward. It is also the design that embarrasses these models: a slope
fitted on fourteen years and run out twenty-four months compounds a straight line
into territory no data supports, and it looks *best* on smooth synthetic data,
which is exactly when a reviewer should trust it least. Instead the profile is
anchored to how the state's recent level sits against its own seasonal mean, and
the strength of that anchoring is fitted rather than assumed. This is what damped
trend methods do, and why damping exists.

**Where the calendar comes from.** The ``Forecaster`` protocol deliberately hands
models a tensor with no index -- a model that cannot see the date cannot cheat on
time. Rather than add a time feature to the shared pipeline for this one model,
the month is recovered from the ``season_sin`` / ``season_cos`` columns that are
already there (the encoding is invertible through ``atan2``) and the state from
its one-hot column. Nothing about :func:`~src.features.build_features` changes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import Config
from src.features import TARGET_LEVEL_COLUMN, FeatureSpec, flatten
from src.models import ForecasterFactory

#: The two columns that encode position within the year.
SEASON_COLUMNS = ("season_sin", "season_cos")

#: Prefix marking a state's one-hot column.
STATE_PREFIX = "state_is_"


class SeasonalError(RuntimeError):
    """Raised when the seasonal model cannot find the columns it needs."""


def months_from_cyclic(sin_values: np.ndarray, cos_values: np.ndarray, period: int) -> np.ndarray:
    """Recover the position within the year from its sine/cosine encoding.

    The encoding is ``(sin, cos)`` of ``2*pi*position/period``, so ``atan2``
    inverts it exactly. Recovering it costs nothing and means this model needs no
    new feature and no change to the shared pipeline.

    Returns:
        Integer position in ``[0, period)``, one per row.
    """
    angle = np.arctan2(sin_values, cos_values)
    position = np.rint(angle / (2.0 * np.pi) * period).astype(int)
    return np.mod(position, period)


@dataclass(frozen=True)
class SeasonalProfile:
    """What one state's year looks like, and how much it varies.

    Attributes:
        level: Mean transformed target per position in the year.
        low: Offset from :attr:`level` to the lower bound, per position.
        high: Offset to the upper bound, per position. Asymmetric on purpose --
            case-rate seasons are right-skewed, and forcing a symmetric band
            around the mean would understate the bad years and overstate the
            quiet ones.
        observed: How many years contributed to each position.
        baseline: The state's overall mean, the anchor is measured against.
        n_observations: Rows the profile was fitted on.
    """

    level: np.ndarray
    low: np.ndarray
    high: np.ndarray
    observed: np.ndarray
    baseline: float
    n_observations: int


class SeasonalTrend:
    """Per-state seasonal profile, anchored to the recent level.

    Satisfies the ``Forecaster`` protocol, so ``run_experiment`` scores it on the
    same rolling-origin folds as everything else. There is no bespoke evaluation
    path and no special-casing in the harness.

    The model is deliberately small: one mean per (state, position-in-year), plus
    a single fitted coefficient controlling how far the recent level pulls that
    mean. A larger model would fit the fourteen years better and project no more
    honestly.
    """

    def __init__(
        self,
        season_indices: tuple[int, int],
        state_indices: dict[str, int],
        anchor_index: int,
        cfg: Config,
    ) -> None:
        self.season_indices = season_indices
        self.state_indices = state_indices
        self.anchor_index = anchor_index
        self.cfg = cfg
        self.label = "seasonal_trend"
        self.profiles_: dict[str, SeasonalProfile] = {}
        self.anchor_weight_: float = 0.0
        self.global_: SeasonalProfile | None = None

    # -- protocol ----------------------------------------------------------- #

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> SeasonalTrend:
        """Fit one profile per state, then one shared anchor coefficient.

        The anchor is fitted globally rather than per state on purpose: it is one
        number estimated from every row, where a per-state version would be one
        number per state estimated from a fourteenth of the data.
        """
        del validation  # nothing to early-stop
        flat = flatten(X)
        period = int(self.cfg.project.seasonal_period)
        positions = self._positions(flat, period)
        states = self._states(flat)
        anchors = flat[:, self.anchor_index]

        alpha = float(self.cfg.conformal.alpha)
        self.global_ = _profile(y, positions, period, float(np.mean(y)), alpha)
        for state in self.state_indices:
            rows = states == state
            if not rows.any():
                continue
            self.profiles_[state] = _profile(
                y[rows], positions[rows], period, float(np.mean(y[rows])), alpha
            )

        self.anchor_weight_ = (
            _fit_anchor(y, self._expected(positions, states), anchors, self._baselines(states))
            if self.cfg.seasonal.use_level_anchor
            else 0.0
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """The state's profile for this position in the year, pulled by the anchor."""
        if self.global_ is None:
            raise SeasonalError("predict called before fit")
        flat = flatten(X)
        period = int(self.cfg.project.seasonal_period)
        positions = self._positions(flat, period)
        states = self._states(flat)

        expected = self._expected(positions, states)
        deviation = flat[:, self.anchor_index] - self._baselines(states)
        return expected + self.anchor_weight_ * deviation

    # -- internals ---------------------------------------------------------- #

    def _positions(self, flat: np.ndarray, period: int) -> np.ndarray:
        sin_index, cos_index = self.season_indices
        return months_from_cyclic(flat[:, sin_index], flat[:, cos_index], period)

    def _states(self, flat: np.ndarray) -> np.ndarray:
        """Which state each row belongs to, from its one-hot column."""
        names = list(self.state_indices)
        block = flat[:, [self.state_indices[name] for name in names]]
        return np.asarray(names, dtype=object)[np.argmax(block, axis=1)]

    def _expected(self, positions: np.ndarray, states: np.ndarray) -> np.ndarray:
        """The profile value for each row, falling back to the pooled profile."""
        assert self.global_ is not None
        out = self.global_.level[positions]
        for state, profile in self.profiles_.items():
            rows = states == state
            if rows.any():
                out[rows] = profile.level[positions[rows]]
        return out

    def _baselines(self, states: np.ndarray) -> np.ndarray:
        """Each row's state mean, which the anchor deviation is measured against."""
        assert self.global_ is not None
        out = np.full(len(states), self.global_.baseline, dtype=float)
        for state, profile in self.profiles_.items():
            rows = states == state
            if rows.any():
                out[rows] = profile.baseline
        return out

    def profile_for(self, state: str) -> SeasonalProfile:
        """One state's fitted profile, falling back to the pooled one."""
        profile = self.profiles_.get(state) or self.global_
        if profile is None:
            raise SeasonalError("profile_for called before fit")
        return profile


def _profile(
    values: np.ndarray,
    positions: np.ndarray,
    period: int,
    baseline: float,
    alpha: float = 0.2,
) -> SeasonalProfile:
    """Mean and observed spread per position in the year, with pooled fallbacks.

    The band comes from **empirical quantiles of this state's own history at that
    position** -- the spread of its actual Septembers -- not from the LSTM's
    conformal residuals and not from a normal assumption. With a decade of data
    that is around ten values per month, which is coarse; it is still a more
    honest statement about September than a symmetric band fitted to the year.

    A position never observed falls back to the pooled spread rather than to
    zero. Zero is a real case rate on this scale, so an unobserved September must
    not silently become a confident prediction of no dengue.
    """
    level = np.full(period, baseline, dtype=float)
    low = np.zeros(period, dtype=float)
    high = np.zeros(period, dtype=float)
    observed = np.zeros(period, dtype=int)

    centred = values - baseline
    pooled_low = float(np.quantile(centred, alpha / 2.0)) if len(values) > 1 else 0.0
    pooled_high = float(np.quantile(centred, 1.0 - alpha / 2.0)) if len(values) > 1 else 0.0

    for position in range(period):
        rows = positions == position
        count = int(rows.sum())
        observed[position] = count
        if count == 0:
            low[position], high[position] = pooled_low, pooled_high
            continue
        here = values[rows]
        level[position] = float(np.mean(here))
        if count > 1:
            residual = here - level[position]
            low[position] = float(np.quantile(residual, alpha / 2.0))
            high[position] = float(np.quantile(residual, 1.0 - alpha / 2.0))
        else:
            low[position], high[position] = pooled_low, pooled_high

    # A band must bracket the value it is a band around. Empirical quantiles do
    # that naturally, but a position whose years agree exactly leaves residuals
    # that are floating-point dust rather than zero, and an offset of +1e-17 puts
    # the lower bound above the estimate it is bounding.
    return SeasonalProfile(
        level=level, low=np.minimum(low, 0.0), high=np.maximum(high, 0.0),
        observed=observed, baseline=baseline, n_observations=len(values),
    )


def _fit_anchor(
    y: np.ndarray, expected: np.ndarray, anchors: np.ndarray, baselines: np.ndarray
) -> float:
    """How strongly the recent level pulls the seasonal mean.

    One least-squares coefficient of the profile's error on the anchor's own
    deviation. Guarded against a degenerate denominator: a state whose level never
    moves gives a zero-variance anchor, and dividing by it would turn a constant
    series into infinities rather than into the obvious answer of "no pull".
    """
    deviation = anchors - baselines
    denominator = float(np.dot(deviation, deviation))
    if not np.isfinite(denominator) or denominator <= 0.0:
        return 0.0
    return float(np.dot(deviation, y - expected) / denominator)


def seasonal_trend(spec: FeatureSpec, cfg: Config) -> ForecasterFactory:
    """Build the seasonal forecaster for one feature set.

    Raises:
        SeasonalError: the feature set carries no cyclic encoding, no state
            identity, or no target lag to anchor on. All three are legitimate
            ablation configurations, so the caller records a skip rather than
            treating it as a failure.
    """
    columns = spec.columns
    missing = [name for name in SEASON_COLUMNS if name not in columns]
    if missing:
        raise SeasonalError(
            f"the seasonal model needs {missing}; set features.cyclic_seasonality: "
            "true. This is a seasonal profile -- without a calendar there is "
            "nothing for it to profile."
        )
    states = {
        name[len(STATE_PREFIX):]: position
        for position, name in enumerate(columns)
        if name.startswith(STATE_PREFIX)
    }
    if not states:
        raise SeasonalError(
            "the seasonal model needs per-state identity columns; set "
            "features.state_identity: true"
        )
    anchor = f"{TARGET_LEVEL_COLUMN}_lag_0"
    if anchor not in columns:
        raise SeasonalError(
            f"the seasonal model anchors on {anchor!r}, which this feature set "
            "does not carry. Set features.include_target_lags: true, or turn "
            "seasonal.use_level_anchor off for pure climatology."
        )

    season = (columns.index(SEASON_COLUMNS[0]), columns.index(SEASON_COLUMNS[1]))
    return lambda: SeasonalTrend(season, states, columns.index(anchor), cfg)


__all__ = [
    "SEASON_COLUMNS",
    "SeasonalError",
    "SeasonalProfile",
    "SeasonalProjection",
    "SeasonalTrend",
    "climatology_year",
    "months_from_cyclic",
    "project_seasonal",
    "seasonal_trend",
]


# --------------------------------------------------------------------------- #
# Projection: answering a month with no covariates for it
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SeasonalProjection:
    """One month's answer from the seasonal profile.

    Deliberately not a :class:`~src.simulate.ForecastStep`. That type carries
    ``mode`` of direct-or-recursive and a reliability that decays with recursion
    depth, neither of which describes this. Sharing the type would invite the
    interface to present the two identically, which is the one thing this model
    must never allow.

    Attributes:
        trend_shift: How much of the answer came from the fitted trend rather
            than from the seasonal profile, on the reported scale. Surfaced so a
            reader can see when a projection is being carried by extrapolation.
        reliability: How well-supported this position in the year is, as the
            fraction of the fitting window that actually observed it. It does
            **not** decay with distance -- a profile two years out is exactly as
            well-supported as one three months out, which is the property that
            justifies the longer reach. What it gives up is elsewhere: this is a
            pattern, not a conditional forecast.
    """

    state: str
    target_date: pd.Timestamp
    predicted_cases_per_100k: float
    lower_cases_per_100k: float
    upper_cases_per_100k: float
    reliability: float
    years_observed: int
    anchored: bool
    trend_shift: float
    periods_ahead: int


def project_seasonal(
    panel: pd.DataFrame, state: str, target_date: pd.Timestamp, cfg: Config
) -> SeasonalProjection:
    """What a typical month looks like for one state, given how the year is running.

    Fitted straight from the panel's target series rather than through
    :func:`~src.features.build_features`, because the whole point is that it needs
    **no covariates at the target date**. That is what lets it answer two years
    out where the recursive path cannot answer seven months out.

    Args:
        panel: Cleaned wide panel.
        state: State to project.
        target_date: The month in question. Only its position in the year and, if
            anchoring is on, the state's recent level affect the answer.
        cfg: Loaded configuration.

    Returns:
        The projection, on the reported case-rate scale.

    Raises:
        SeasonalError: the state has no observed history to profile.
    """
    from src.features import inverse_target_transform, target_level

    levels = target_level(panel, cfg).dropna()
    try:
        series = levels.loc[state]
    except KeyError as exc:
        raise SeasonalError(f"{state!r} is not in the panel") from exc
    if series.empty:
        raise SeasonalError(f"{state!r} has no observed history to profile")

    window = _trailing(series, cfg)
    period = int(cfg.project.seasonal_period)
    positions = _positions_of(window.index, period)
    profile = _profile(
        window.to_numpy(), positions, period,
        float(window.mean()), float(cfg.conformal.alpha),
    )

    slot = _positions_of(pd.DatetimeIndex([pd.Timestamp(target_date)]), period)[0]
    centre = float(profile.level[slot])

    ahead = _periods_ahead(window.index[-1], pd.Timestamp(target_date), period)

    # Past the anchored window this becomes pure climatology, and it degrades
    # itself rather than trusting the caller to ask correctly. How this year is
    # running says nothing about a month seven years out, and a trend
    # extrapolated that far is arithmetic rather than evidence. What survives is
    # the profile -- "an August here looks like this" -- which is as true of 2030
    # as of next year, because it is a claim about Augusts and not about 2030.
    anchored = ahead <= int(cfg.seasonal.max_projection_periods)
    level, slope = (
        _level_and_trend(window, profile, period, cfg) if anchored else (0.0, 0.0)
    )
    drift = slope * _damped_steps(ahead, float(cfg.seasonal.trend_damping))
    centre += level + drift

    anchored = anchored and bool(cfg.seasonal.use_level_anchor)

    low = centre + float(profile.low[slot])
    high = centre + float(profile.high[slot])
    observed = int(profile.observed[slot])

    return SeasonalProjection(
        state=state,
        target_date=pd.Timestamp(target_date),
        predicted_cases_per_100k=float(inverse_target_transform(np.asarray(centre), cfg)),
        lower_cases_per_100k=float(inverse_target_transform(np.asarray(low), cfg)),
        upper_cases_per_100k=float(inverse_target_transform(np.asarray(high), cfg)),
        reliability=min(1.0, observed / max(cfg.seasonal.trailing_years, 1)),
        years_observed=observed,
        anchored=anchored,
        trend_shift=float(inverse_target_transform(np.asarray(centre), cfg))
        - float(inverse_target_transform(np.asarray(centre - drift), cfg)),
        periods_ahead=ahead,
    )


def _trailing(series: pd.Series, cfg: Config) -> pd.Series:
    """The last ``seasonal.trailing_years`` of a state's history.

    Restricting the window is a judgement that a decade ago is less like next
    year than last year is. Everything available is used when the state is
    younger than the window, rather than refusing -- a short series gives a
    weaker profile, and :attr:`SeasonalProjection.reliability` reports that.
    """
    periods = int(cfg.project.seasonal_period) * int(cfg.seasonal.trailing_years)
    return series.iloc[-periods:] if len(series) > periods else series


def _periods_ahead(last: pd.Timestamp, target: pd.Timestamp, period: int) -> int:
    """How many periods separate the end of the fitting window from the target."""
    if period == 12:
        return (target.year - last.year) * 12 + (target.month - last.month)
    return int((pd.Timestamp(target) - pd.Timestamp(last)).days // (365 // period))


def _positions_of(index: pd.DatetimeIndex, period: int) -> np.ndarray:
    """Position within the year for each date, matching the cyclic encoding."""
    if period == 12:
        return np.asarray(index.month - 1, dtype=int)
    return np.asarray((index.dayofyear - 1) * period // 366, dtype=int)


def _level_and_trend(
    window: pd.Series, profile: SeasonalProfile, period: int, cfg: Config
) -> tuple[float, float]:
    """Fit how the state sits against its own profile, and where that is heading.

    One regression, deseasonalised: strip each observation of its month's typical
    value, then fit ``residual = level + slope * periods_from_the_end``. The
    intercept is the level anchor -- how this year is running against a typical
    year -- and the slope is the long-run drift over the trailing window. Fitting
    them together rather than as two mechanisms means the anchor is not counting
    drift that the trend already explains.

    Returns:
        ``(level, slope)`` on the transformed target scale, both zero when the
        window is too short or the corresponding config switch is off.
    """
    if window.empty:
        return 0.0, 0.0

    positions = _positions_of(pd.DatetimeIndex(window.index), period)
    residual = window.to_numpy() - profile.level[positions]
    if not cfg.seasonal.use_level_anchor and not cfg.seasonal.use_trend:
        return 0.0, 0.0

    steps = np.arange(len(residual), dtype=float) - (len(residual) - 1)
    level = float(np.mean(residual))
    slope = 0.0

    if cfg.seasonal.use_trend and len(residual) > period:
        spread = float(np.dot(steps, steps))
        if spread > 0.0:
            centred = steps - steps.mean()
            denominator = float(np.dot(centred, centred))
            if denominator > 0.0:
                slope = float(np.dot(centred, residual - residual.mean()) / denominator)
                level = float(residual.mean() - slope * steps.mean())

    if not cfg.seasonal.use_level_anchor:
        level = 0.0
    return level, slope


def _damped_steps(steps: int, damping: float) -> float:
    """How many periods of drift a projection ``steps`` ahead actually gets.

    ``phi + phi^2 + ... + phi^steps``. At ``damping = 1`` this is just ``steps``,
    a straight line extrapolated as far as asked -- which is how these models
    embarrass themselves at two years. Below 1 it converges to
    ``phi / (1 - phi)``, so the trend bends toward a ceiling instead of running
    away, and the projection makes a weaker claim the further out it reaches.
    """
    if steps <= 0:
        return 0.0
    if damping >= 1.0:
        return float(steps)
    if damping <= 0.0:
        return 0.0
    return float(damping * (1.0 - damping**steps) / (1.0 - damping))


def climatology_year(
    panel: pd.DataFrame, state: str, cfg: Config
) -> pd.DataFrame:
    """One state's typical year: every position in the cycle, with its spread.

    What to draw when the question is about a month years away. A time series
    running from the last observation out to 2030 would be eighty months of the
    same repeated shape, which says nothing and buries the history that gives it
    weight. The typical year says exactly what the profile knows.

    Returns:
        Columns ``position``, ``label``, ``predicted``, ``lower``, ``upper`` and
        ``observed`` (how many years contributed), one row per position.
    """
    from src.features import inverse_target_transform, target_level

    levels = target_level(panel, cfg).dropna()
    try:
        series = levels.loc[state]
    except KeyError as exc:
        raise SeasonalError(f"{state!r} is not in the panel") from exc
    if series.empty:
        raise SeasonalError(f"{state!r} has no observed history to profile")

    window = _trailing(series, cfg)
    period = int(cfg.project.seasonal_period)
    profile = _profile(
        window.to_numpy(),
        _positions_of(pd.DatetimeIndex(window.index), period),
        period,
        float(window.mean()),
        float(cfg.conformal.alpha),
    )

    def rate(values: np.ndarray) -> np.ndarray:
        return np.asarray(inverse_target_transform(values, cfg), dtype=float)

    labels = (
        [pd.Timestamp(2000, position + 1, 1).strftime("%b") for position in range(period)]
        if period == 12
        else [str(position + 1) for position in range(period)]
    )
    return pd.DataFrame(
        {
            "position": np.arange(period),
            "label": labels,
            "predicted": rate(profile.level),
            "lower": rate(profile.level + profile.low),
            "upper": rate(profile.level + profile.high),
            "observed": profile.observed,
        }
    )
