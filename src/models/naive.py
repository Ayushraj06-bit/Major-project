"""Baselines the LSTM has to beat.

These exist to answer "what does good mean?" before anything complex is built. If
the LSTM cannot beat seasonal-naive, that is a finding worth having in month two
rather than month six.

Each model is a handful of lines, because the work is already done: the target
level and its lags are features, so persistence and seasonal-naive are lookups
rather than calculations. All three satisfy the ``Forecaster`` protocol, so
``run_experiment`` runs them through exactly the same code path as the LSTM.

Column positions come from the :class:`~src.features.FeatureSpec` and are supplied
to the constructor by the factory functions at the bottom of this module. That is
what keeps the protocol free of a spec argument, and keeps ``run_experiment`` free
of any branch on model type.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from src.config import Config
from src.features import TARGET_LEVEL_COLUMN, FeatureSpec, flatten
from src.models import Forecaster, ForecasterFactory


class BaselineError(RuntimeError):
    """Raised when a baseline cannot find the column it needs."""


class LastValue:
    """Predict by carrying one feature column forward unchanged.

    Both naive baselines are this, differing only in which column they read:
    persistence reads the current value, seasonal-naive reads the value from the
    same period a year earlier. Writing it once and configuring the column keeps
    them from drifting apart.

    Nothing is learned. ``fit`` records how many rows it saw purely so a run
    record shows the baseline was exercised on the same folds as everything else.
    """

    def __init__(self, column_index: int, label: str) -> None:
        self.column_index = column_index
        self.label = label
        self.n_fitted_ = 0

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> LastValue:
        """Record the training size. There are no parameters to estimate."""
        self.n_fitted_ = len(X)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Read the chosen column at the forecast origin, the last timestep."""
        return X[:, -1, self.column_index].astype(float)


class GBMBaseline:
    """Gradient boosting on the flattened feature view.

    The bar a neural network actually has to clear. Tree ensembles are strong on
    tabular problems of this size, need no scaling, and train in seconds — so if
    the LSTM cannot beat this either, the sequence structure is not earning its
    keep and the report should say so.
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.model_: Any | None = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> GBMBaseline:
        """Fit on the 2-D view of the window. Needs no validation block."""
        from sklearn.ensemble import HistGradientBoostingRegressor

        gbm = self.cfg.model.gbm
        model = HistGradientBoostingRegressor(
            max_iter=gbm.n_estimators,
            max_depth=gbm.max_depth,
            learning_rate=gbm.learning_rate,
            random_state=self.cfg.project.seed,
        )
        model.fit(flatten(X), y)
        self.model_ = model
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict from the same 2-D view."""
        if self.model_ is None:
            raise RuntimeError("GBMBaseline.predict called before fit")
        return np.asarray(self.model_.predict(flatten(X)), dtype=float)


class RidgeBaseline:
    """Ridge regression on the flattened window.

    README section 6 asks for a linear baseline alongside gradient boosting, and
    it earns its place: on short series a regularised linear model is frequently
    competitive with anything deeper, and it is the cheapest way to find out
    whether a problem needs non-linearity at all.

    Ridge rather than ordinary least squares because the flattened design matrix
    is wide and highly collinear — every lag of a variable correlates with its
    neighbours — and unregularised coefficients on that are numerically unstable
    and uninterpretable.
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.model_: Any | None = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> RidgeBaseline:
        """Fit with the regularisation strength chosen by internal cross-validation.

        ``RidgeCV`` picks alpha from the training fold only, so the choice never
        sees the test period.
        """
        from sklearn.linear_model import RidgeCV

        model = RidgeCV(alphas=self.cfg.model.ridge.alphas)
        model.fit(flatten(X), y)
        self.model_ = model
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict from the same 2-D view."""
        if self.model_ is None:
            raise RuntimeError("RidgeBaseline.predict called before fit")
        return np.asarray(self.model_.predict(flatten(X)), dtype=float)


# --------------------------------------------------------------------------- #
# Factories — where the FeatureSpec dependency lives
# --------------------------------------------------------------------------- #


def persistence(spec: FeatureSpec) -> ForecasterFactory:
    """Carry the last observed value forward.

    Reads ``target_level_lag_0``: the case rate observed at the forecast origin, on
    the same transformed scale as the target. Because the target sits at
    ``t + horizon`` with ``horizon >= 1``, this is information genuinely available
    at prediction time.
    """
    index = _column_index(spec, f"{TARGET_LEVEL_COLUMN}_lag_0")
    return lambda: LastValue(index, "persistence")


def seasonal_naive(spec: FeatureSpec, cfg: Config) -> ForecasterFactory:
    """Predict the value observed in the same period a year earlier.

    The lag is ``seasonal_period - horizon``, not ``seasonal_period``. Predicting
    ``y(t + h)`` with "the same period last year" means ``y(t + h - P)``, which seen
    from the origin at ``t`` is lag ``P - h``. Using lag ``P`` would instead predict
    a year before the *origin*, which is a different and weaker baseline.
    """
    offset = cfg.project.seasonal_period - spec.horizon
    if offset <= 0:
        raise BaselineError(
            f"horizon {spec.horizon} is not shorter than the seasonal period "
            f"{cfg.project.seasonal_period}; a seasonal-naive forecast is undefined"
        )
    index = _column_index(spec, f"{TARGET_LEVEL_COLUMN}_lag_{offset}")
    return lambda: LastValue(index, "seasonal_naive")


def gradient_boosting(cfg: Config) -> ForecasterFactory:
    """Gradient boosting on the flat view. Needs no scaling wrapper."""
    return lambda: GBMBaseline(cfg)


def ridge(cfg: Config) -> ForecasterFactory:
    """Ridge regression on the flat view.

    Wrapped in the fold scaler: a penalised linear model is not scale-invariant,
    so unscaled inputs would penalise large-valued features far more heavily than
    small-valued ones for no reason connected to their usefulness.
    """
    from src.models.scaling import StandardScaled

    return lambda: StandardScaled(RidgeBaseline(cfg))


def _seasonal_trend(spec: FeatureSpec, cfg: Config) -> ForecasterFactory:
    """The seasonal-profile forecaster. Imported here to keep the module acyclic."""
    from src.models.seasonal import seasonal_trend

    return seasonal_trend(spec, cfg)


def _column_index(spec: FeatureSpec, column: str) -> int:
    """Position of a required column, with an actionable error when absent."""
    try:
        return spec.columns.index(column)
    except ValueError:
        raise BaselineError(
            f"baseline needs feature column {column!r}, which is not in this "
            f"FeatureSpec. Set features.include_target_lags: true and keep 'cases' "
            f"in features.sources. Available target columns: "
            f"{[c for c in spec.columns if c.startswith(TARGET_LEVEL_COLUMN)]}"
        ) from None


def baseline_builders(
    spec: FeatureSpec, cfg: Config
) -> dict[str, Callable[[], ForecasterFactory]]:
    """Deferred constructors for every baseline, keyed by run name.

    Deferred because a baseline can be legitimately unavailable: the climate-only
    ablation carries no case history, so persistence has nothing to carry forward
    and raises on construction. A caller running a grid needs to record that as a
    skip rather than have the whole grid die.

    One place to add a baseline, so the Phase 4 comparison and the ablation table
    stay in step.
    """
    return {
        "persistence": lambda: persistence(spec),
        "seasonal_naive": lambda: seasonal_naive(spec, cfg),
        "gbm": lambda: gradient_boosting(cfg),
        "ridge": lambda: ridge(cfg),
        # A different kind of model from the three above: a seasonal profile
        # rather than a lookup or a regression on the flat view. It lives in
        # src/models/seasonal.py because it is long enough to warrant its own
        # module, and is registered here so one place still lists every model the
        # comparison and the ablation table run.
        "seasonal_trend": lambda: _seasonal_trend(spec, cfg),
    }


def baseline_factories(spec: FeatureSpec, cfg: Config) -> dict[str, ForecasterFactory]:
    """Every baseline, built. Raises if any is unavailable for this feature set."""
    return {name: build() for name, build in baseline_builders(spec, cfg).items()}


__all__ = [
    "BaselineError",
    "GBMBaseline",
    "LastValue",
    "RidgeBaseline",
    "baseline_builders",
    "baseline_factories",
    "gradient_boosting",
    "persistence",
    "ridge",
    "seasonal_naive",
]


# Static protocol conformance: these must satisfy Forecaster without inheriting it.
_: type[Forecaster] = LastValue
__: type[Forecaster] = GBMBaseline
___: type[Forecaster] = RidgeBaseline
