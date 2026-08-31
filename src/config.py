"""Typed, validated configuration.

``config.yaml`` is the single source of every tunable in the project. It is read
once by :func:`load_config` into a frozen :class:`Config` tree and never touched
again — nothing downstream re-reads the file or hard-codes a value.

Validation happens at construction time, so a bad config fails at startup with a
keyed message rather than fifty minutes into a training loop. Missing keys and
unknown keys are both errors: a typo in ``config.yaml`` cannot silently fall back
to a default.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from datetime import date
from functools import cache
from pathlib import Path
from types import UnionType
from typing import Any, TypeVar, Union, get_args, get_origin, get_type_hints

import yaml

CONFIG_PATH_ENV_VAR = "DENGUE_CONFIG"
DEFAULT_CONFIG_FILENAME = "config.yaml"

#: Periods per year for each supported granularity. Also the seasonal-naive lag.
SEASONAL_PERIODS: dict[str, int] = {"monthly": 12, "weekly": 52}

VALID_TARGET_TRANSFORMS = frozenset({"log1p", "none"})
VALID_SOURCES = frozenset({"climate", "cases", "demographic", "awareness"})
VALID_SPLIT_SCHEMES = frozenset({"rolling_origin"})
VALID_ALERT_BASES = frozenset({"upper", "point"})
VALID_CLAMP_STRATEGIES = frozenset({"sigma", "minmax"})
VALID_INTERPOLATION_METHODS = frozenset({"linear", "time", "nearest"})
VALID_SPATIAL_WEIGHTS = frozenset({"uniform", "population"})
VALID_ROLLING_AGGS = frozenset({"mean", "sum", "min", "max", "std"})
VALID_EXPLAINERS = frozenset({"kernel", "gradient"})
VALID_RISK_METHODS = frozenset({"quantile", "ewma"})

_T = TypeVar("_T")


class ConfigError(ValueError):
    """Raised when ``config.yaml`` is malformed, incomplete, or self-inconsistent."""


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ProjectConfig:
    """Identity and global determinism settings."""

    name: str
    seed: int
    granularity: str

    def __post_init__(self) -> None:
        if self.granularity not in SEASONAL_PERIODS:
            raise ConfigError(
                f"project.granularity: expected one of {sorted(SEASONAL_PERIODS)}, "
                f"got {self.granularity!r}"
            )
        if self.seed < 0:
            raise ConfigError(f"project.seed: must be non-negative, got {self.seed}")

    @property
    def seasonal_period(self) -> int:
        """Periods per year — the lag a seasonal-naive baseline reads from."""
        return SEASONAL_PERIODS[self.granularity]


@dataclass(frozen=True)
class Paths:
    """Filesystem layout. All paths are absolute once the config is loaded."""

    data_raw: Path
    data_interim: Path
    data_processed: Path
    results: Path
    runs: Path
    figures: Path
    metrics: Path


@dataclass(frozen=True)
class DataConfig:
    """What slice of the world the model is trained on, and the target definition."""

    sources: tuple[str, ...]
    start_date: date
    end_date: date
    states: tuple[str, ...]
    target_column: str
    target_transform: str
    population_normalisation: int

    def __post_init__(self) -> None:
        _require_known_sources(self.sources, "data.sources")
        if self.start_date >= self.end_date:
            raise ConfigError(
                f"data: start_date ({self.start_date}) must precede end_date ({self.end_date})"
            )
        if not self.states:
            raise ConfigError("data.states: must list at least one state")
        duplicates = _duplicates(self.states)
        if duplicates:
            raise ConfigError(f"data.states: duplicate entries {duplicates}")
        if self.target_transform not in VALID_TARGET_TRANSFORMS:
            raise ConfigError(
                f"data.target_transform: expected one of {sorted(VALID_TARGET_TRANSFORMS)}, "
                f"got {self.target_transform!r}"
            )
        if self.population_normalisation <= 0:
            raise ConfigError(
                "data.population_normalisation: must be positive, "
                f"got {self.population_normalisation}"
            )


@dataclass(frozen=True)
class FeatureConfig:
    """Which derived features ``build_features`` emits.

    Narrowing ``sources`` is how the data-source ablation (README §6, configs
    A/B/C) is expressed — the ablation is configuration, not a code branch.
    """

    sources: tuple[str, ...]
    include_lags: bool
    include_spatial: bool
    include_target_lags: bool
    include_state_identity: bool
    level_variables: tuple[str, ...]
    lags: tuple[int, ...]
    lag_variables: tuple[str, ...]
    rolling_windows: tuple[int, ...]
    rolling_variables: tuple[str, ...]
    rolling_aggs: tuple[str, ...]
    spatial_lags: tuple[int, ...]
    spatial_variable: str
    spatial_weight_scheme: str
    static_variables: tuple[str, ...]
    sequence_length: int
    cyclic_seasonality: bool
    selected_columns: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_positive_ascending(self.lags, "features.lags")
        _require_positive_ascending(self.rolling_windows, "features.rolling_windows")
        _require_positive_ascending(self.spatial_lags, "features.spatial_lags")
        if self.sequence_length <= 0:
            raise ConfigError(
                f"features.sequence_length: must be positive, got {self.sequence_length}"
            )
        _require_known_sources(self.sources, "features.sources")
        if self.spatial_weight_scheme not in VALID_SPATIAL_WEIGHTS:
            raise ConfigError(
                "features.spatial_weight_scheme: expected one of "
                f"{sorted(VALID_SPATIAL_WEIGHTS)}, got {self.spatial_weight_scheme!r}"
            )
        unknown_aggs = sorted(set(self.rolling_aggs) - VALID_ROLLING_AGGS)
        if unknown_aggs:
            raise ConfigError(
                f"features.rolling_aggs: unknown aggregation(s) {unknown_aggs}, "
                f"expected {sorted(VALID_ROLLING_AGGS)}"
            )
        if not self.rolling_aggs:
            raise ConfigError("features.rolling_aggs: must list at least one aggregation")
        overlap = sorted(set(self.level_variables) & set(self.static_variables))
        if overlap:
            raise ConfigError(
                f"features: {overlap} appear in both level_variables and static_variables, "
                "which would build the same column twice"
            )
        for name, values in (
            ("features.level_variables", self.level_variables),
            ("features.lag_variables", self.lag_variables),
            ("features.rolling_variables", self.rolling_variables),
            ("features.static_variables", self.static_variables),
            ("features.rolling_aggs", self.rolling_aggs),
        ):
            duplicates = _duplicates(values)
            if duplicates:
                raise ConfigError(f"{name}: duplicate entries {duplicates}")


@dataclass(frozen=True)
class LSTMConfig:
    """LSTM hyperparameters. Deliberately small — see brain.md D-14."""

    units: int
    layers: int
    dropout: float
    recurrent_dropout: float
    learning_rate: float
    batch_size: int
    max_epochs: int
    early_stopping_patience: int
    validation_fraction: float
    state_embedding_dim: int

    def __post_init__(self) -> None:
        _require_positive(self.units, "model.lstm.units")
        _require_positive(self.layers, "model.lstm.layers")
        if self.layers > 2:
            raise ConfigError(
                f"model.lstm.layers: {self.layers} is too deep for a dataset this "
                "size; one or two layers only (brain.md D-14)"
            )
        _require_positive(self.batch_size, "model.lstm.batch_size")
        _require_positive(self.max_epochs, "model.lstm.max_epochs")
        _require_positive(self.early_stopping_patience, "model.lstm.early_stopping_patience")
        _require_positive(self.state_embedding_dim, "model.lstm.state_embedding_dim")
        _require_positive(self.learning_rate, "model.lstm.learning_rate")
        _require_unit_interval(self.dropout, "model.lstm.dropout")
        _require_unit_interval(self.recurrent_dropout, "model.lstm.recurrent_dropout")
        _require_unit_interval(
            self.validation_fraction, "model.lstm.validation_fraction", exclusive_low=True
        )
        if self.early_stopping_patience >= self.max_epochs:
            raise ConfigError(
                "model.lstm.early_stopping_patience "
                f"({self.early_stopping_patience}) must be less than max_epochs "
                f"({self.max_epochs}), otherwise early stopping can never trigger"
            )


@dataclass(frozen=True)
class GBMConfig:
    """Gradient-boosting baseline hyperparameters (README §6)."""

    n_estimators: int
    max_depth: int
    learning_rate: float

    def __post_init__(self) -> None:
        _require_positive(self.n_estimators, "model.gbm.n_estimators")
        _require_positive(self.max_depth, "model.gbm.max_depth")
        _require_positive(self.learning_rate, "model.gbm.learning_rate")


@dataclass(frozen=True)
class RidgeConfig:
    """Linear baseline hyperparameters (README section 6)."""

    alphas: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.alphas:
            raise ConfigError("model.ridge.alphas: must list at least one value")
        if any(alpha <= 0 for alpha in self.alphas):
            raise ConfigError(
                f"model.ridge.alphas: all values must be positive, got {list(self.alphas)}"
            )


@dataclass(frozen=True)
class ModelConfig:
    """Hyperparameters for every learned model in the ablation."""

    lstm: LSTMConfig
    gbm: GBMConfig
    ridge: RidgeConfig


@dataclass(frozen=True)
class SplitConfig:
    """Rolling-origin cross-validation geometry (brain.md D-03, D-04).

    There is deliberately no shuffle option. Random splitting leaks future
    information into training and invalidates every downstream result.
    """

    scheme: str
    n_folds: int
    initial_train_size: int
    test_size: int
    step: int

    def __post_init__(self) -> None:
        if self.scheme not in VALID_SPLIT_SCHEMES:
            raise ConfigError(
                f"split.scheme: expected one of {sorted(VALID_SPLIT_SCHEMES)}, "
                f"got {self.scheme!r}"
            )
        if self.n_folds < 2:
            raise ConfigError(
                f"split.n_folds: must be at least 2 to report mean +/- std, got {self.n_folds}"
            )
        _require_positive(self.initial_train_size, "split.initial_train_size")
        _require_positive(self.test_size, "split.test_size")
        _require_positive(self.step, "split.step")


@dataclass(frozen=True)
class ForecastConfig:
    """Forecast lead times, in periods of ``project.granularity``."""

    horizons: tuple[int, ...]
    max_recursive_steps: int

    def __post_init__(self) -> None:
        _require_positive_ascending(self.horizons, "forecast.horizons")
        _require_positive(self.max_recursive_steps, "forecast.max_recursive_steps")


@dataclass(frozen=True)
class SeasonalConfig:
    """The seasonal-profile forecaster's settings.

    A different model from the LSTM, with a different reach and different
    weaknesses, so it gets its own section rather than extending ``forecast``.
    """

    trailing_years: int
    use_level_anchor: bool
    use_trend: bool
    trend_damping: float
    max_projection_periods: int
    climatology_max_years: int

    def __post_init__(self) -> None:
        _require_positive(self.trailing_years, "seasonal.trailing_years")
        _require_positive(
            self.max_projection_periods, "seasonal.max_projection_periods"
        )
        _require_positive(
            self.climatology_max_years, "seasonal.climatology_max_years"
        )
        if not 0.0 <= self.trend_damping <= 1.0:
            raise ConfigError(
                "seasonal.trend_damping: must be between 0 (no trend) and 1 "
                f"(undamped straight line), got {self.trend_damping}"
            )


@dataclass(frozen=True)
class ConformalConfig:
    """Distribution-free prediction intervals (brain.md S-2, A-2b)."""

    alpha: float
    min_calibration_residuals: int

    def __post_init__(self) -> None:
        _require_unit_interval(self.alpha, "conformal.alpha", exclusive_low=True)
        _require_positive(
            self.min_calibration_residuals, "conformal.min_calibration_residuals"
        )

    @property
    def coverage(self) -> float:
        """Nominal interval coverage, e.g. 0.8 for ``alpha = 0.2``."""
        return 1.0 - self.alpha


@dataclass(frozen=True)
class ActionSet:
    """The actions a domain expert has attached to one risk tier.

    A typed entry rather than a free-form mapping, so a missing tier or a typo in
    a tier name fails at startup instead of producing a recommendation with no
    actions attached to it.
    """

    tier: str
    actions: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.tier:
            raise ConfigError("risk.actions: every entry needs a tier")
        if not self.actions:
            raise ConfigError(f"risk.actions[{self.tier}]: must list at least one action")


@dataclass(frozen=True)
class RiskConfig:
    """Risk banding for the decision-support layer (brain.md D-10, D-11).

    ``quantiles`` are positions in the historical case distribution, so the
    thresholds are derived from data rather than chosen by hand. ``tiers`` names
    the bands they cut, and so must be one longer than ``quantiles``.
    """

    method: str
    quantiles: tuple[float, ...]
    tiers: tuple[str, ...]
    alert_on: str
    ewma_alpha: float
    ewma_sigma: float
    actions: tuple[ActionSet, ...]
    action_source: str

    def __post_init__(self) -> None:
        if self.method not in VALID_RISK_METHODS:
            raise ConfigError(
                f"risk.method: expected one of {sorted(VALID_RISK_METHODS)}, "
                f"got {self.method!r}"
            )
        _require_unit_interval(self.ewma_alpha, "risk.ewma_alpha", exclusive_low=True)
        _require_positive(self.ewma_sigma, "risk.ewma_sigma")
        self._validate_every_tier_has_actions()
        if not self.quantiles:
            raise ConfigError("risk.quantiles: must list at least one quantile")
        for value in self.quantiles:
            _require_unit_interval(value, "risk.quantiles", exclusive_low=True)
        if list(self.quantiles) != sorted(set(self.quantiles)):
            raise ConfigError(
                f"risk.quantiles: must be strictly ascending and unique, got {list(self.quantiles)}"
            )
        if len(self.tiers) != len(self.quantiles) + 1:
            raise ConfigError(
                f"risk.tiers: {len(self.quantiles)} quantile(s) cut "
                f"{len(self.quantiles) + 1} band(s), but {len(self.tiers)} tier name(s) given"
            )
        duplicates = _duplicates(self.tiers)
        if duplicates:
            raise ConfigError(f"risk.tiers: duplicate entries {duplicates}")
        if self.alert_on not in VALID_ALERT_BASES:
            raise ConfigError(
                f"risk.alert_on: expected one of {sorted(VALID_ALERT_BASES)}, "
                f"got {self.alert_on!r}"
            )

    def _validate_every_tier_has_actions(self) -> None:
        """A tier with no actions would produce a recommendation recommending nothing."""
        mapped = [entry.tier for entry in self.actions]
        duplicates = _duplicates(tuple(mapped))
        if duplicates:
            raise ConfigError(f"risk.actions: duplicate tier(s) {duplicates}")

        missing = sorted(set(self.tiers) - set(mapped))
        if missing:
            raise ConfigError(f"risk.actions: no actions defined for tier(s) {missing}")
        unknown = sorted(set(mapped) - set(self.tiers))
        if unknown:
            raise ConfigError(
                f"risk.actions: tier(s) {unknown} are not in risk.tiers {list(self.tiers)}"
            )

    def actions_for(self, tier: str) -> tuple[str, ...]:
        """The configured actions for one tier."""
        for entry in self.actions:
            if entry.tier == tier:
                return entry.actions
        raise ConfigError(f"risk.actions: no entry for tier {tier!r}")


@dataclass(frozen=True)
class SimulateConfig:
    """Guardrails for scenario simulation (brain.md D-09)."""

    clamp_strategy: str
    clamp_n_sigma: float

    def __post_init__(self) -> None:
        if self.clamp_strategy not in VALID_CLAMP_STRATEGIES:
            raise ConfigError(
                f"simulate.clamp_strategy: expected one of {sorted(VALID_CLAMP_STRATEGIES)}, "
                f"got {self.clamp_strategy!r}"
            )
        _require_positive(self.clamp_n_sigma, "simulate.clamp_n_sigma")


@dataclass(frozen=True)
class ExplainConfig:
    """SHAP budget. KernelExplainer is slow; these bound the cost."""

    background_samples: int
    nsamples: int
    top_k_features: int
    max_explained_rows: int
    explainer: str

    def __post_init__(self) -> None:
        _require_positive(self.background_samples, "explain.background_samples")
        _require_positive(self.nsamples, "explain.nsamples")
        _require_positive(self.top_k_features, "explain.top_k_features")
        _require_positive(self.max_explained_rows, "explain.max_explained_rows")
        if self.explainer not in VALID_EXPLAINERS:
            raise ConfigError(
                f"explain.explainer: expected one of {sorted(VALID_EXPLAINERS)}, "
                f"got {self.explainer!r}"
            )


@dataclass(frozen=True)
class QualityConfig:
    """Thresholds for the data-quality report."""

    outlier_mad_threshold: float
    min_coverage: float
    min_periods_for_per_state_model: int

    def __post_init__(self) -> None:
        _require_positive(self.outlier_mad_threshold, "quality.outlier_mad_threshold")
        _require_unit_interval(self.min_coverage, "quality.min_coverage", exclusive_low=True)
        _require_positive(
            self.min_periods_for_per_state_model, "quality.min_periods_for_per_state_model"
        )


@dataclass(frozen=True)
class PreprocessConfig:
    """Gap-filling policy. Deliberately narrow — see :mod:`src.preprocess`."""

    interpolate_variables: tuple[str, ...]
    interpolation_method: str
    max_interpolation_gap: int

    def __post_init__(self) -> None:
        if self.interpolation_method not in VALID_INTERPOLATION_METHODS:
            raise ConfigError(
                "preprocess.interpolation_method: expected one of "
                f"{sorted(VALID_INTERPOLATION_METHODS)}, got {self.interpolation_method!r}"
            )
        _require_positive(self.max_interpolation_gap, "preprocess.max_interpolation_gap")
        duplicates = _duplicates(self.interpolate_variables)
        if duplicates:
            raise ConfigError(f"preprocess.interpolate_variables: duplicate entries {duplicates}")


@dataclass(frozen=True)
class ProductionConfig:
    """How the frozen production model is chosen and fitted."""

    model: str
    selection_metric: str
    parsimony_tiebreak: bool
    calibration_periods: int

    def __post_init__(self) -> None:
        if not self.model:
            raise ConfigError("production.model: must name a model")
        if not self.selection_metric:
            raise ConfigError("production.selection_metric: must name a metric")
        _require_positive(self.calibration_periods, "production.calibration_periods")


@dataclass(frozen=True)
class ExperimentSpec:
    """One named ablation configuration.

    Only the fields that differ from the base config are given; anything omitted
    is inherited. That keeps the grid readable and makes each entry state exactly
    what it is testing.
    """

    name: str
    sources: tuple[str, ...] | None = None
    include_lags: bool | None = None
    include_spatial: bool | None = None
    include_target_lags: bool | None = None
    note: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ConfigError("experiments: every entry needs a name")
        if self.sources is not None:
            _require_known_sources(self.sources, f"experiments.{self.name}.sources")


@dataclass(frozen=True)
class Config:
    """The whole configuration tree. Frozen, validated, loaded once."""

    project: ProjectConfig
    paths: Paths
    data: DataConfig
    features: FeatureConfig
    model: ModelConfig
    split: SplitConfig
    forecast: ForecastConfig
    seasonal: SeasonalConfig
    conformal: ConformalConfig
    risk: RiskConfig
    simulate: SimulateConfig
    explain: ExplainConfig
    quality: QualityConfig
    preprocess: PreprocessConfig
    production: ProductionConfig
    experiments: tuple[ExperimentSpec, ...]

    def __post_init__(self) -> None:
        self._validate_experiment_names_are_unique()
        self._validate_seasonal_lag_present()
        self._validate_ablation_is_a_subset_of_what_is_loaded()
        self._validate_horizons_fit_test_window()

    def _validate_seasonal_lag_present(self) -> None:
        """A seasonal-naive baseline reads the same period last year out of ``X``.

        If that lag is not among the emitted features the baseline cannot be
        evaluated through the shared ``Forecaster`` protocol, and the ablation
        stops being one loop (brain.md A-2c).
        """
        period = self.project.seasonal_period
        if period not in self.lags:
            raise ConfigError(
                f"features.lags must include the seasonal period ({period}) for "
                f"{self.project.granularity!r} data, so the seasonal-naive baseline can read "
                f"its input from X. Got {list(self.lags)}."
            )

    def _validate_experiment_names_are_unique(self) -> None:
        """Duplicate names would overwrite each other's saved runs."""
        duplicates = _duplicates(tuple(item.name for item in self.experiments))
        if duplicates:
            raise ConfigError(f"experiments: duplicate name(s) {duplicates}")

    def _validate_ablation_is_a_subset_of_what_is_loaded(self) -> None:
        """Model inputs can only come from sources that were actually fetched."""
        extra = sorted(set(self.features.sources) - set(self.data.sources))
        if extra:
            raise ConfigError(
                f"features.sources {extra} are not in data.sources {list(self.data.sources)}; "
                "a source cannot feed the model without being loaded"
            )

    def _validate_horizons_fit_test_window(self) -> None:
        """A horizon longer than the test window cannot be scored in any fold."""
        longest = max(self.forecast.horizons)
        if longest > self.split.test_size:
            raise ConfigError(
                f"forecast.horizons: longest horizon ({longest}) exceeds split.test_size "
                f"({self.split.test_size}), so no fold can evaluate it"
            )

    @property
    def lags(self) -> tuple[int, ...]:
        """Shorthand for ``config.features.lags``."""
        return self.features.lags

    @property
    def seasonal_period(self) -> int:
        """Shorthand for ``config.project.seasonal_period``."""
        return self.project.seasonal_period


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def resolve_config_path(path: Path | str | None = None) -> Path:
    """Return the config file to load.

    Precedence: explicit argument, then ``$DENGUE_CONFIG``, then ``config.yaml``
    beside the project root. Tests point at a fixture by setting the env var.
    """
    if path is not None:
        return Path(path)
    from_env = os.environ.get(CONFIG_PATH_ENV_VAR)
    if from_env:
        return Path(from_env)
    return _project_root() / DEFAULT_CONFIG_FILENAME


@cache
def load_config(path: Path | str | None = None) -> Config:
    """Read, validate and cache the configuration tree.

    Cached per resolved path, so the file is parsed once per process however many
    modules ask for it. Call :func:`clear_config_cache` in tests that rewrite the
    file between assertions.

    Raises:
        ConfigError: the file is missing, unparseable, has missing or unknown
            keys, has a value of the wrong type, or fails a validation rule.
    """
    config_path = resolve_config_path(path).resolve()
    if not config_path.is_file():
        raise ConfigError(f"config file not found: {config_path}")

    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{config_path}: could not parse YAML: {exc}") from exc

    if not isinstance(raw, Mapping):
        raise ConfigError(f"{config_path}: expected a top-level mapping, got {type(raw).__name__}")

    raw = _absolutise_paths(raw, root=config_path.parent)
    return _build(Config, raw, path="config")


def clear_config_cache() -> None:
    """Drop the memoised config. Only needed by tests."""
    load_config.cache_clear()


# --------------------------------------------------------------------------- #
# Generic construction — one builder, not one parser per section
# --------------------------------------------------------------------------- #


def _build(cls: type[_T], raw: Any, path: str) -> _T:
    """Construct a frozen config dataclass from a mapping, recursing into nested ones.

    Missing and unknown keys are both errors, reported with their dotted path so
    the message names the offending line in ``config.yaml``.
    """
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{path}: expected a mapping, got {type(raw).__name__}")

    import dataclasses

    hints = get_type_hints(cls)
    all_fields = fields(cls)  # type: ignore[arg-type]
    expected = {field.name for field in all_fields}
    # A field carrying a default is optional in the YAML; one without is required.
    required = {
        field.name
        for field in all_fields
        if field.default is dataclasses.MISSING
        and field.default_factory is dataclasses.MISSING
    }
    provided = set(raw)

    missing = sorted(required - provided)
    if missing:
        raise ConfigError(f"{path}: missing key(s) {missing}")
    unknown = sorted(provided - expected)
    if unknown:
        raise ConfigError(f"{path}: unknown key(s) {unknown}; expected {sorted(expected)}")

    kwargs = {
        name: _coerce(hints[name], raw[name], f"{path}.{name}")
        for name in expected
        if name in provided
    }
    return cls(**kwargs)


def _coerce(hint: Any, value: Any, path: str) -> Any:
    """Convert one YAML scalar or collection to the type its field declares."""
    # `X | None` declares an optional override; None passes through, anything else
    # is coerced to X.
    if get_origin(hint) in (Union, UnionType):
        options = [a for a in get_args(hint) if a is not type(None)]
        if value is None:
            return None
        if len(options) == 1:
            return _coerce(options[0], value, path)

    # isinstance(hint, type) distinguishes a nested dataclass *class* from an
    # instance of one; only the class can be constructed from a mapping.
    if isinstance(hint, type) and is_dataclass(hint):
        return _build(hint, value, path)

    if get_origin(hint) is tuple:
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{path}: expected a list, got {type(value).__name__}")
        item_hint = get_args(hint)[0]
        return tuple(_coerce(item_hint, item, f"{path}[{i}]") for i, item in enumerate(value))

    if hint is Path:
        if not isinstance(value, (str, Path)):
            raise ConfigError(f"{path}: expected a path string, got {type(value).__name__}")
        return Path(value)

    if hint is date:
        if isinstance(value, date):
            return value
        try:
            return date.fromisoformat(str(value))
        except ValueError as exc:
            raise ConfigError(f"{path}: expected an ISO date (YYYY-MM-DD), got {value!r}") from exc

    if hint is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{path}: expected a boolean, got {type(value).__name__}")
        return value

    # bool is a subclass of int, so it must be rejected explicitly.
    if hint is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{path}: expected an integer, got {value!r}")
        return value

    if hint is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{path}: expected a number, got {value!r}")
        return float(value)

    if hint is str:
        if not isinstance(value, str):
            raise ConfigError(f"{path}: expected a string, got {type(value).__name__}")
        return value

    raise ConfigError(f"{path}: unsupported field type {hint!r}")


def _absolutise_paths(raw: Mapping[str, Any], root: Path) -> dict[str, Any]:
    """Resolve the ``paths`` section against the config file's directory.

    Done on the raw mapping rather than after construction so that :class:`Paths`
    can stay frozen and every consumer sees absolute paths from the start.
    """
    result = dict(raw)
    paths = result.get("paths")
    if isinstance(paths, Mapping):
        result["paths"] = {
            key: str(value if Path(str(value)).is_absolute() else root / str(value))
            for key, value in paths.items()
        }
    return result


# --------------------------------------------------------------------------- #
# Validation helpers
# --------------------------------------------------------------------------- #


def _project_root() -> Path:
    """The repository root — the parent of the ``src`` package."""
    return Path(__file__).resolve().parent.parent


def _duplicates(values: tuple[Any, ...]) -> list[Any]:
    """Return the values appearing more than once, in first-seen order."""
    seen: set[Any] = set()
    repeated: list[Any] = []
    for value in values:
        if value in seen and value not in repeated:
            repeated.append(value)
        seen.add(value)
    return repeated


def _require_positive(value: float, path: str) -> None:
    """Raise unless ``value`` is strictly greater than zero."""
    if value <= 0:
        raise ConfigError(f"{path}: must be positive, got {value}")


def _require_unit_interval(value: float, path: str, *, exclusive_low: bool = False) -> None:
    """Raise unless ``value`` lies in [0, 1) — or (0, 1) when ``exclusive_low``."""
    low_ok = value > 0 if exclusive_low else value >= 0
    if not (low_ok and value < 1):
        bound = "(0, 1)" if exclusive_low else "[0, 1)"
        raise ConfigError(f"{path}: must lie in {bound}, got {value}")


def _require_known_sources(values: tuple[str, ...], path: str) -> None:
    """Raise unless every entry is a recognised, non-duplicated source name."""
    if not values:
        raise ConfigError(f"{path}: must list at least one source")
    unknown = sorted(set(values) - VALID_SOURCES)
    if unknown:
        raise ConfigError(f"{path}: unknown source(s) {unknown}, expected {sorted(VALID_SOURCES)}")
    duplicates = _duplicates(values)
    if duplicates:
        raise ConfigError(f"{path}: duplicate entries {duplicates}")


def _require_positive_ascending(values: tuple[int, ...], path: str) -> None:
    """Raise unless ``values`` is a non-empty, strictly ascending list of positives."""
    if not values:
        raise ConfigError(f"{path}: must list at least one value")
    if any(value <= 0 for value in values):
        raise ConfigError(f"{path}: all values must be positive, got {list(values)}")
    if list(values) != sorted(set(values)):
        raise ConfigError(f"{path}: must be strictly ascending and unique, got {list(values)}")
