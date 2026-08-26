"""Feature engineering: the one function training, evaluation and simulation share.

:func:`build_features` is **pure**. Same panel and config in, same arrays out; no
global state, no cached results, nothing fitted to the data. That is not a style
preference — the Phase 8 scenario simulator works by modifying the raw panel and
calling straight back into this function. If anything here remembered state
between calls, or estimated a parameter from the data, a simulated scenario would
be transformed differently from the training data and every what-if result would
be quietly wrong.

The same rule keeps scaling out. No ``StandardScaler``, no ``MinMaxScaler``, no
encoder. Scaling is fitted per fold inside :mod:`src.evaluate`. Everything built
here is either a fixed function of config or a function of each row's own past.

Ablations are config, not code. Configurations A/B/C narrow ``features.sources``;
D/E toggle ``features.include_lags``; F toggles ``features.include_spatial``.
There is exactly one feature builder, and every ablation is a different argument
to it.

Column names are never hardcoded downstream. :class:`FeatureSpec` records the
order of the feature axis and, for every derived column, which raw variable it
came from and how. Phase 8 uses it to find every column that must move when a
driver changes; Phase 7 uses it to sum SHAP attributions back to raw variables.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.config import Config
from src.panel import PanelError
from src.sources import PANEL_KEYS
from src.sources.base import discover_sources
from src.sources.registry import adjacency, normalise_state

#: Name of the constructed target column.
TARGET_COLUMN = "target"

#: Name of the unshifted target level, and the stem of its autoregressive lags.
TARGET_LEVEL_COLUMN = "target_level"

#: Transform kinds recorded in :class:`FeatureOrigin`.
TRANSFORM_IDENTITY = "identity"
TRANSFORM_LAG = "lag"
TRANSFORM_ROLLING = "rolling"
TRANSFORM_SPATIAL_LAG = "spatial_lag"
TRANSFORM_CYCLIC = "cyclic"
TRANSFORM_STATIC = "static"

#: Pseudo-variable for calendar features, which derive from no raw source.
CALENDAR_VARIABLE = "__calendar__"

#: Pseudo-variable for state-identity features, which derive from no raw source.
STATE_VARIABLE = "__state__"

#: Prefix of the one-hot state-identity columns.
STATE_PREFIX = "state_is_"


class FeatureError(RuntimeError):
    """Raised when the panel cannot produce the configured features."""


@dataclass(frozen=True)
class FeatureOrigin:
    """Where one derived column came from.

    ``raw_variable`` is the panel column it ultimately depends on, which is what
    the simulator and the explainer both key off — not the derived name.
    """

    column: str
    raw_variable: str
    transform: str
    lag: int | None = None
    window: int | None = None
    agg: str | None = None


@dataclass(frozen=True)
class FeatureSpec:
    """Everything needed to interpret the arrays :func:`build_features` returns.

    Attributes:
        columns: Feature-axis names, in the order they occupy in ``X``.
        origins: Per-column provenance, keyed by column name.
        sample_index: ``(state, date)`` of each sample, where ``date`` is the last
            observed period in that sample's window — the forecast origin, not the
            period being predicted. Rolling-origin splitting and the dashboard's
            state mapping both need this, so it travels with the arrays rather
            than being recomputed.
        timesteps: Window length on the second axis of ``X``.
        horizon: Periods ahead the target is taken from.
        target_name: Human-readable description of the target.
        dropped_states: Configured states that produced no usable sample. Recorded
            rather than left silent: a state whose neighbours are all outside the
            study has no spatial lag, so every one of its windows is incomplete and
            it disappears from the dataset entirely. That also makes the spatial
            ablation unfair unless it is noticed, because configurations E and F
            would then be scored on different sets of states.
    """

    columns: tuple[str, ...]
    origins: Mapping[str, FeatureOrigin]
    sample_index: pd.MultiIndex
    timesteps: int
    horizon: int
    target_name: str
    dropped_states: tuple[str, ...] = ()

    @property
    def n_features(self) -> int:
        """Size of the feature axis."""
        return len(self.columns)

    @property
    def flat_columns(self) -> tuple[str, ...]:
        """Names for the flattened 2-D view, in ``X.reshape(n, -1)`` order.

        Formatted ``<column>@t-<k>``, where ``t-0`` is the forecast origin.
        """
        return tuple(
            f"{column}@t-{self.timesteps - 1 - step}"
            for step in range(self.timesteps)
            for column in self.columns
        )

    def raw_variable_of(self, column: str) -> str:
        """The raw panel variable a derived column depends on.

        Accepts either a feature-axis name or a flattened ``name@t-k`` name, so
        SHAP output over the flat view maps back without the caller unpicking the
        suffix itself.
        """
        base = column.split("@", 1)[0]
        try:
            return self.origins[base].raw_variable
        except KeyError:
            raise KeyError(
                f"unknown feature column {column!r}; known columns are {list(self.columns)}"
            ) from None

    def columns_from(self, raw_variable: str) -> tuple[str, ...]:
        """Every derived column that depends on one raw variable.

        The simulator's correctness rests on this. Raising rainfall must move
        ``rainfall``, every ``rainfall_lag_k``, and every rolling aggregate of it
        together; changing one and not the others produces an input the model
        never saw in training.
        """
        return tuple(
            column for column in self.columns if self.origins[column].raw_variable == raw_variable
        )

    @property
    def raw_variables(self) -> tuple[str, ...]:
        """Distinct raw variables feeding the model, calendar terms excluded."""
        seen = {origin.raw_variable for origin in self.origins.values()}
        return tuple(sorted(seen - {CALENDAR_VARIABLE, STATE_VARIABLE}))

    def column_indices(self, *, transform: str | None = None,
                       raw_variable: str | None = None) -> tuple[int, ...]:
        """Positions of columns matching a transform kind and/or raw variable.

        Lets a model route columns by what they *are* rather than by hardcoded
        names, so adding a static feature needs no change in the model.
        """
        return tuple(
            position
            for position, column in enumerate(self.columns)
            if (transform is None or self.origins[column].transform == transform)
            and (raw_variable is None or self.origins[column].raw_variable == raw_variable)
        )

    @property
    def state_columns(self) -> tuple[int, ...]:
        """Positions of the one-hot state-identity columns."""
        return self.column_indices(raw_variable=STATE_VARIABLE)

    @property
    def static_columns(self) -> tuple[int, ...]:
        """Positions of static columns, state identity excluded."""
        return tuple(
            position
            for position in self.column_indices(transform=TRANSFORM_STATIC)
            if position not in set(self.state_columns)
        )

    @property
    def sequence_columns(self) -> tuple[int, ...]:
        """Positions of the genuinely time-varying columns."""
        excluded = set(self.static_columns) | set(self.state_columns)
        return tuple(p for p in range(len(self.columns)) if p not in excluded)

    def to_dict(self) -> dict[str, object]:
        """Serialise to plain JSON types, for the production artifact.

        The spec has to survive a round trip through disk or the frozen model is
        uninterpretable: without the column order, nothing downstream can build an
        input the model will accept, and without the origins the simulator cannot
        tell which columns move when a driver changes.
        """
        return {
            "columns": list(self.columns),
            "origins": [dataclasses.asdict(self.origins[column]) for column in self.columns],
            "sample_index": [
                [str(state), pd.Timestamp(date).isoformat()]
                for state, date in self.sample_index
            ],
            "timesteps": self.timesteps,
            "horizon": self.horizon,
            "target_name": self.target_name,
            "dropped_states": list(self.dropped_states),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> FeatureSpec:
        """Rebuild a spec saved by :meth:`to_dict`."""
        origins = {
            entry["column"]: FeatureOrigin(**entry) for entry in payload["origins"]
        }
        pairs = [(state, pd.Timestamp(date)) for state, date in payload["sample_index"]]
        return cls(
            columns=tuple(payload["columns"]),
            origins=origins,
            sample_index=pd.MultiIndex.from_tuples(pairs, names=list(PANEL_KEYS)),
            timesteps=int(payload["timesteps"]),
            horizon=int(payload["horizon"]),
            target_name=str(payload["target_name"]),
            dropped_states=tuple(payload.get("dropped_states", ())),
        )

    def group_by_raw_variable(self, values: np.ndarray) -> dict[str, float]:
        """Sum per-flat-column values (such as SHAP attributions) by raw variable.

        Args:
            values: One value per entry of :attr:`flat_columns`.

        Returns:
            Raw variable to summed value. A ``timesteps x features`` attribution
            grid is unreadable; this is what the report and dashboard show.
        """
        flat = self.flat_columns
        if len(values) != len(flat):
            raise ValueError(f"expected {len(flat)} values, got {len(values)}")
        totals: dict[str, float] = {}
        for name, value in zip(flat, values, strict=True):
            variable = self.raw_variable_of(name)
            totals[variable] = totals.get(variable, 0.0) + float(value)
        return totals


def build_features(
    panel: pd.DataFrame, cfg: Config, *, horizon: int | None = None
) -> tuple[np.ndarray, np.ndarray, FeatureSpec]:
    """Turn a cleaned panel into model-ready arrays.

    Pure: no global state, no fitted objects, and ``panel`` is not mutated.

    Args:
        panel: Cleaned wide panel indexed by ``(state, date)``.
        cfg: Loaded configuration. Ablations are expressed entirely through it.
        horizon: Forecast lead time. Defaults to the first entry of
            ``forecast.horizons``; passed explicitly when sweeping horizons, since
            one call builds one horizon.

    Returns:
        ``X`` of shape ``(n_samples, timesteps, n_features)``, ``y`` of shape
        ``(n_samples,)`` on the transformed scale, and the :class:`FeatureSpec`
        describing them.

    Raises:
        FeatureError: a configured variable is absent from the panel, or no sample
            survives the window and target requirements.
    """
    horizon = horizon if horizon is not None else cfg.forecast.horizons[0]
    _require_panel(panel)

    frame = panel.copy()
    available = _selected_variables(frame, cfg)

    features: dict[str, pd.Series] = {}
    origins: dict[str, FeatureOrigin] = {}

    levels = _intersect(cfg.features.level_variables, available)
    _collect(features, origins, add_identity(frame, levels))
    if cfg.features.include_lags:
        _collect(
            features,
            origins,
            add_lags(frame, _intersect(cfg.features.lag_variables, available), cfg.features.lags),
        )
        _collect(
            features,
            origins,
            add_rolling(
                frame,
                _intersect(cfg.features.rolling_variables, available),
                cfg.features.rolling_windows,
                cfg.features.rolling_aggs,
            ),
        )
    if cfg.features.include_target_lags and cfg.data.target_column in available:
        _collect(features, origins, add_target_lags(frame, cfg, horizon))
    if cfg.features.include_spatial:
        _collect(features, origins, add_spatial_lags(frame, cfg))
    if cfg.features.cyclic_seasonality:
        _collect(features, origins, add_cyclic(frame, cfg))
    if cfg.features.include_state_identity:
        _collect(features, origins, add_state_identity(frame))
    statics = _intersect(cfg.features.static_variables, available)
    _collect(features, origins, add_static(frame, statics))

    if not features:
        raise FeatureError(
            "no features were built; check features.sources and the include_* flags"
        )

    features, origins = apply_selection(features, origins, cfg)
    design = pd.DataFrame(features, index=frame.index)
    target = build_target(frame, cfg, horizon)
    return window_sequences(design, target, cfg, horizon, origins)


# --------------------------------------------------------------------------- #
# Generic transformations, each driven by config
# --------------------------------------------------------------------------- #


def add_identity(
    frame: pd.DataFrame, columns: Sequence[str]
) -> list[tuple[pd.Series, FeatureOrigin]]:
    """Pass selected raw variables through unchanged, as the value at ``t``.

    Driven by ``features.level_variables`` rather than by everything the selected
    sources happen to provide. ``population`` is a case in point: it is loaded so
    the target can be expressed as a rate, but it is not a driver, and including
    it here would put the target's own denominator into the design matrix.
    """
    return [
        (frame[column].rename(column), FeatureOrigin(column, column, TRANSFORM_IDENTITY, lag=0))
        for column in columns
    ]


def add_lags(
    frame: pd.DataFrame,
    columns: Sequence[str],
    lags: Sequence[int],
    group_key: str = "state",
) -> list[tuple[pd.Series, FeatureOrigin]]:
    """Shift each column back by each lag, within each state.

    One generic function, not one per variable: adding a lagged driver is an edit
    to ``features.lag_variables``, never to this module.

    Grouping by state is what stops Kerala's series being shifted into Odisha's at
    the boundary between them in the index.
    """
    built: list[tuple[pd.Series, FeatureOrigin]] = []
    for column in columns:
        grouped = frame[column].groupby(level=group_key, sort=False)
        for lag in lags:
            name = f"{column}_lag_{lag}"
            built.append(
                (
                    grouped.shift(lag).rename(name),
                    FeatureOrigin(name, column, TRANSFORM_LAG, lag=lag),
                )
            )
    return built


def add_rolling(
    frame: pd.DataFrame,
    columns: Sequence[str],
    windows: Sequence[int],
    aggs: Sequence[str],
    group_key: str = "state",
) -> list[tuple[pd.Series, FeatureOrigin]]:
    """Trailing rolling aggregates, within each state.

    Windows are trailing and inclusive of ``t``, which is an observed value at
    prediction time, so no future information enters. ``min_periods`` equals the
    window: a partial window at the start of a series would be a different
    statistic wearing the same column name.
    """
    built: list[tuple[pd.Series, FeatureOrigin]] = []
    for column in columns:
        grouped = frame[column].groupby(level=group_key, sort=False)
        for window in windows:
            rolled = grouped.rolling(window=window, min_periods=window)
            for agg in aggs:
                name = f"{column}_roll{window}_{agg}"
                series = getattr(rolled, agg)().droplevel(0).rename(name)
                built.append(
                    (
                        series.reindex(frame.index),
                        FeatureOrigin(name, column, TRANSFORM_ROLLING, window=window, agg=agg),
                    )
                )
    return built


def add_spatial_lags(frame: pd.DataFrame, cfg: Config) -> list[tuple[pd.Series, FeatureOrigin]]:
    """Neighbouring states' lagged values, weighted by a configured scheme.

    Dengue travels with people, so a neighbour's outbreak last period is genuine
    signal — and this is what makes the project's "spatio-temporal" claim true
    rather than aspirational.

    Weight schemes:
        ``uniform``: every land neighbour counts equally. The default, because it
            depends on nothing but the fixed adjacency graph.
        ``population``: neighbours weighted by their share of neighbouring
            population, using each state's mean population over the panel. That
            mean is computed across the whole window, including periods that will
            fall in a test fold. It is a slowly-varying denominator rather than a
            target-derived quantity, so the risk is small — but it is not nothing,
            which is why it is not the default.

    A state with no neighbours in the study — an island, or one whose neighbours
    were not loaded — yields NaN rather than zero. Zero would assert that
    neighbouring case counts were observed to be zero, which is a different claim
    from not having observed them.
    """
    variable = cfg.features.spatial_variable
    if variable not in frame.columns:
        raise FeatureError(
            f"features.spatial_variable {variable!r} is not in the panel; "
            f"available columns are {list(frame.columns)}"
        )

    wide = frame[variable].unstack(level="state")
    weights = spatial_weights(frame, cfg)

    built: list[tuple[pd.Series, FeatureOrigin]] = []
    for lag in cfg.features.spatial_lags:
        lagged = wide.shift(lag)
        neighbour_values = pd.DataFrame(
            {
                state: (
                    lagged[list(state_weights)].mul(list(state_weights.values()), axis=1).sum(
                        axis=1, min_count=1
                    )
                    if state_weights
                    else pd.Series(np.nan, index=lagged.index)
                )
                for state, state_weights in weights.items()
            }
        )
        # Naming the column axis is what lets stack() produce a (date, state)
        # MultiIndex that can be reordered to the panel's (state, date).
        neighbour_values.columns.name = "state"
        name = f"{variable}_spatial_lag_{lag}"
        stacked = neighbour_values.stack(future_stack=True)
        series = stacked.reorder_levels(list(PANEL_KEYS)).rename(name)
        built.append(
            (
                series.reindex(frame.index),
                FeatureOrigin(name, variable, TRANSFORM_SPATIAL_LAG, lag=lag),
            )
        )
    return built


def spatial_weights(frame: pd.DataFrame, cfg: Config) -> dict[str, dict[str, float]]:
    """Normalised neighbour weights per state, restricted to states in the panel.

    Restricting to loaded states is the honest behaviour: a neighbour outside the
    study contributes nothing because its case counts were never observed.
    """
    states = list(frame.index.get_level_values("state").unique())
    graph = adjacency()
    scheme = cfg.features.spatial_weight_scheme

    if scheme == "population":
        if "population" not in frame.columns:
            raise FeatureError(
                "features.spatial_weight_scheme is 'population' but the panel has no "
                "'population' column; add the demographic source to data.sources"
            )
        mass = frame["population"].groupby(level="state").mean()
    else:
        mass = pd.Series(1.0, index=states)

    weights: dict[str, dict[str, float]] = {}
    for state in states:
        neighbours = sorted(graph[normalise_state(state)] & set(states))
        total = float(sum(mass.get(neighbour, 0.0) for neighbour in neighbours))
        weights[state] = (
            {neighbour: float(mass.get(neighbour, 0.0)) / total for neighbour in neighbours}
            if neighbours and total > 0
            else {}
        )
    return weights


def add_cyclic(frame: pd.DataFrame, cfg: Config) -> list[tuple[pd.Series, FeatureOrigin]]:
    """Sine and cosine of position within the year.

    A pair, not a single index: month 12 and month 1 are adjacent, and only the
    sin/cos pair encodes that without a discontinuity every December. Captures the
    annual rhythm that climate alone misses — reporting cycles, vector-control
    calendars, behaviour.
    """
    period = float(cfg.project.seasonal_period)
    dates = frame.index.get_level_values("date")
    position = _position_in_year(dates, cfg.project.granularity)
    angle = 2.0 * np.pi * position / period

    return [
        (
            pd.Series(np.sin(angle), index=frame.index, name="season_sin"),
            FeatureOrigin("season_sin", CALENDAR_VARIABLE, TRANSFORM_CYCLIC),
        ),
        (
            pd.Series(np.cos(angle), index=frame.index, name="season_cos"),
            FeatureOrigin("season_cos", CALENDAR_VARIABLE, TRANSFORM_CYCLIC),
        ),
    ]


def add_static(
    frame: pd.DataFrame, columns: Sequence[str]
) -> list[tuple[pd.Series, FeatureOrigin]]:
    """Slowly-varying state attributes, passed through without lagging.

    Lagging population density teaches a sequence model nothing — the value is
    effectively the same at every timestep — so it enters as a level.
    """
    return [
        (frame[column].rename(column), FeatureOrigin(column, column, TRANSFORM_STATIC))
        for column in columns
    ]


def target_level(frame: pd.DataFrame, cfg: Config) -> pd.Series:
    """The target quantity as observed at ``t``, before any forward shift.

    ``log(cases_per_100k + 1)`` by default. The rate makes states comparable in a
    pooled model; the log stabilises variance so a single outbreak period cannot
    dominate the loss.
    """
    missing = [column for column in (cfg.data.target_column, "population") if column not in frame]
    if missing:
        raise FeatureError(
            f"target construction needs column(s) {missing}; the panel has "
            f"{list(frame.columns)}. Keep 'cases' and 'demographic' in data.sources — "
            "features.sources controls model inputs, not what the target is built from."
        )

    rate = frame[cfg.data.target_column] / frame["population"] * cfg.data.population_normalisation
    transformed = np.log1p(rate) if cfg.data.target_transform == "log1p" else rate
    return transformed.rename(TARGET_LEVEL_COLUMN)


def apply_selection(
    features: dict[str, pd.Series],
    origins: dict[str, FeatureOrigin],
    cfg: Config,
) -> tuple[dict[str, pd.Series], dict[str, FeatureOrigin]]:
    """Narrow the design matrix to ``features.selected_columns``, when set.

    This is how SHAP-driven feature selection re-enters the pipeline: the explainer
    ranks columns, the chosen names go into config, and the next run builds only
    those. Selection is therefore configuration like every other ablation, not a
    separate code path.

    State-identity columns are always kept. They are not drivers competing for a
    place in a top-k ranking, they are the pooled model's way of knowing whose
    series it is looking at, and dropping them would silently change the
    architecture rather than the feature set.

    Raises:
        FeatureError: a selected column was never built, which usually means the
            selection came from a run with different ablation flags.
    """
    wanted = set(cfg.features.selected_columns)
    if not wanted:
        return features, origins

    missing = sorted(wanted - set(features))
    if missing:
        raise FeatureError(
            f"features.selected_columns names {missing[:5]} which this configuration "
            "does not build. A selection is only valid for the ablation flags it was "
            "produced under."
        )

    keep = [
        column
        for column in features
        if column in wanted or origins[column].raw_variable == STATE_VARIABLE
    ]
    return (
        {column: features[column] for column in keep},
        {column: origins[column] for column in keep},
    )


def add_state_identity(frame: pd.DataFrame) -> list[tuple[pd.Series, FeatureOrigin]]:
    """One-hot columns naming which state each row belongs to.

    The pooled model needs to know whose series it is looking at. Carried as
    ordinary columns rather than a side channel so the ``Forecaster`` protocol
    stays a two-argument ``fit``, and so tree models can split on state too.

    The LSTM does not feed these to its recurrent layers. It reads them at the
    final timestep and projects them through a bias-free dense layer, which is
    exactly a learned embedding, then concatenates that after the recurrence. A
    state's identity does not change over a window, so putting it in the sequence
    would only waste capacity.
    """
    states = sorted(frame.index.get_level_values("state").unique())
    labels = frame.index.get_level_values("state")
    return [
        (
            pd.Series(
                (labels == state).astype(float), index=frame.index,
                name=f"{STATE_PREFIX}{state}",
            ),
            FeatureOrigin(f"{STATE_PREFIX}{state}", STATE_VARIABLE, TRANSFORM_STATIC),
        )
        for state in states
    ]


def build_target(frame: pd.DataFrame, cfg: Config, horizon: int) -> pd.Series:
    """Shift the target level ``h`` periods forward, within each state.

    The shift is taken per state, so the last periods of one state's series never
    borrow the first periods of the next.
    """
    return (
        target_level(frame, cfg)
        .groupby(level="state", sort=False)
        .shift(-horizon)
        .rename(TARGET_COLUMN)
    )


def target_lag_offsets(cfg: Config, horizon: int) -> tuple[int, ...]:
    """Which lags of the target level to emit as autoregressive features.

    Three groups, unioned:

    * **0** — the currently observed value. This is what a persistence baseline
      predicts, and it is legitimately known at forecast time because the target
      sits at ``t + horizon`` with ``horizon >= 1``.
    * **the configured driver lags** — the target's own history is as useful a
      driver as any climate variable.
    * **``seasonal_period - horizon``** — what a seasonal-naive baseline needs.
      Predicting the value from the same period last year means ``y(t+h-P)``, which
      seen from origin ``t`` is lag ``P - h``, not lag ``P``. Emitting it here
      rather than demanding it in config keeps the baseline runnable whatever
      driver lags are configured.
    """
    period = cfg.project.seasonal_period
    offsets = {0, *cfg.features.lags}
    if period - horizon > 0:
        offsets.add(period - horizon)
    return tuple(sorted(offsets))


def add_target_lags(
    frame: pd.DataFrame, cfg: Config, horizon: int
) -> list[tuple[pd.Series, FeatureOrigin]]:
    """Autoregressive terms, on the target's own transformed scale.

    Distinct from ``cases_lag_k``, which carries raw counts. These are the case
    *rate*, log-transformed exactly as the target is, so a baseline can predict one
    directly and a model can learn a correction to it.

    Provenance points back at the case column, so the simulator moves these
    together with everything else derived from cases.
    """
    level = target_level(frame, cfg)
    grouped = level.groupby(level="state", sort=False)

    built: list[tuple[pd.Series, FeatureOrigin]] = []
    for lag in target_lag_offsets(cfg, horizon):
        name = f"{TARGET_LEVEL_COLUMN}_lag_{lag}"
        series = level if lag == 0 else grouped.shift(lag)
        built.append(
            (
                series.rename(name),
                FeatureOrigin(name, cfg.data.target_column, TRANSFORM_LAG, lag=lag),
            )
        )
    return built


def window_sequences(
    design: pd.DataFrame,
    target: pd.Series,
    cfg: Config,
    horizon: int,
    origins: Mapping[str, FeatureOrigin],
) -> tuple[np.ndarray, np.ndarray, FeatureSpec]:
    """Slide a fixed window over each state's rows to produce ``(n, T, F)``.

    Windows never span two states, and a sample is kept only when its whole window
    and its target are observed. Dropping incomplete windows rather than filling
    them keeps every training sample a real one — the alternative is teaching the
    model to reproduce padding.
    """
    timesteps = cfg.features.sequence_length
    columns = tuple(design.columns)

    blocks: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    keys: list[tuple[str, pd.Timestamp]] = []

    for state, group in design.groupby(level="state", sort=True):
        values = group.to_numpy(dtype=float)
        if len(values) < timesteps:
            continue
        dates = group.index.get_level_values("date")
        # Aligned on the full key rather than sliced by state, so the two can
        # never drift apart if the groupby order and the target order differ.
        state_target = target.reindex(group.index).to_numpy(dtype=float)

        # (n_windows, timesteps, n_features), windows ending at each row from
        # timesteps-1 onward.
        windows = np.lib.stride_tricks.sliding_window_view(
            values, window_shape=timesteps, axis=0
        ).transpose(0, 2, 1)
        ends = np.arange(timesteps - 1, len(values))
        window_target = state_target[ends]

        valid = np.isfinite(windows).all(axis=(1, 2)) & np.isfinite(window_target)
        if not valid.any():
            continue

        blocks.append(windows[valid])
        targets.append(window_target[valid])
        keys.extend((state, dates[end]) for end in ends[valid])

    if not blocks:
        raise FeatureError(
            f"no complete samples: sequence_length={timesteps}, horizon={horizon}, "
            f"max lag={max(cfg.features.lags) if cfg.features.include_lags else 0}. "
            "Each state needs enough consecutive observed periods to fill one window "
            "and still have a target ahead of it."
        )

    sample_index = pd.MultiIndex.from_tuples(keys, names=list(PANEL_KEYS))
    present = set(sample_index.get_level_values("state"))
    dropped = tuple(
        sorted(set(design.index.get_level_values("state").unique()) - present)
    )

    spec = FeatureSpec(
        columns=columns,
        origins={column: origins[column] for column in columns},
        sample_index=sample_index,
        timesteps=timesteps,
        horizon=horizon,
        dropped_states=dropped,
        target_name=(
            f"log(cases per {cfg.data.population_normalisation:,} + 1) at t+{horizon}"
            if cfg.data.target_transform == "log1p"
            else f"cases per {cfg.data.population_normalisation:,} at t+{horizon}"
        ),
    )
    return np.concatenate(blocks), np.concatenate(targets), spec


def flatten(X: np.ndarray) -> np.ndarray:
    """Collapse ``(n, timesteps, features)`` to the 2-D view.

    Tree models and ``KernelExplainer`` both need this; the column order matches
    :attr:`FeatureSpec.flat_columns`.
    """
    if X.ndim != 3:
        raise ValueError(f"expected a 3-D array, got shape {X.shape}")
    return X.reshape(X.shape[0], -1)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _selected_variables(frame: pd.DataFrame, cfg: Config) -> tuple[str, ...]:
    """Panel columns belonging to the sources this ablation selects.

    The source-to-variable mapping comes from the source classes themselves, so
    adding a variable to a source makes it selectable here with no edit.
    """
    owners: dict[str, str] = {}
    for name, source in discover_sources().items():
        for variable in source.variables:
            owners[variable] = name

    selected = set(cfg.features.sources)
    return tuple(
        column
        for column in frame.columns
        if column in owners and owners[column] in selected
    )


def _intersect(requested: Sequence[str], available: Sequence[str]) -> tuple[str, ...]:
    """Requested variables that this ablation actually has, in requested order.

    Silent narrowing is correct here: naming ``cases`` in ``lag_variables`` and
    then running the climate-only ablation should drop the case lags, not fail.
    """
    have = set(available)
    return tuple(column for column in requested if column in have)


def _collect(
    features: dict[str, pd.Series],
    origins: dict[str, FeatureOrigin],
    built: Sequence[tuple[pd.Series, FeatureOrigin]],
) -> None:
    """Accumulate built columns, rejecting any name collision."""
    for series, origin in built:
        if origin.column in features:
            raise FeatureError(f"duplicate feature column {origin.column!r}")
        features[origin.column] = series
        origins[origin.column] = origin


def _position_in_year(dates: pd.Index, granularity: str) -> np.ndarray:
    """Zero-based position of each date within its year, at the given granularity."""
    if granularity == "monthly":
        return dates.month.to_numpy(dtype=float) - 1.0
    if granularity == "weekly":
        return dates.isocalendar().week.to_numpy(dtype=float) - 1.0
    raise FeatureError(f"unsupported granularity {granularity!r}")


def _require_panel(panel: pd.DataFrame) -> None:
    """Reject anything that is not a ``(state, date)``-indexed wide panel."""
    if not isinstance(panel.index, pd.MultiIndex) or tuple(panel.index.names) != PANEL_KEYS:
        raise PanelError(
            f"panel must be indexed by {list(PANEL_KEYS)}, got {list(panel.index.names)}"
        )
    if not panel.index.is_monotonic_increasing:
        raise PanelError(
            "panel index must be sorted by (state, date); lags and windows assume it"
        )