"""Distribution-free prediction intervals by split conformal calibration.

The idea is small: fit the model, look at how wrong it was on data it did not
train on, and use the spread of those residuals as the width of future intervals.
No distributional assumption, no second model, and no hundred forward passes.

Chosen over Monte Carlo dropout for three reasons. It is cheaper, needing one
extra pass over a calibration block rather than a hundred over every prediction.
It is distribution-free, where MC dropout implicitly assumes the dropout ensemble
approximates a posterior. And it adapts: a rolling calibration window tracks the
model's *current* error, so when transmission changes and the model gets worse,
the intervals widen on their own.

The wrapper is itself a :class:`~src.models.Forecaster`, so any model gains
intervals without knowing anything about them, and the naive baselines get them
too. That is what makes interval calibration comparable across the whole ablation
rather than being an LSTM-only feature.

Coverage is the thing to check and is easy to get wrong: an 80% interval that
contains 55% of held-out actuals is not a conservative interval, it is a broken
one. :func:`coverage` measures it and the harness reports it every run.
"""

from __future__ import annotations

import numpy as np

from src.config import Config
from src.models import Forecaster, ForecasterFactory


class UncertaintyError(RuntimeError):
    """Raised when intervals cannot be calibrated or produced."""


class ConformalForecaster:
    """Wrap any forecaster so it also produces prediction intervals.

    Calibration uses the fold's validation block, which :mod:`src.splits` already
    cuts as a genuine future time window sitting between training and test. Using
    training residuals instead would give intervals far too narrow, because the
    model has already seen those points.

    Args:
        base: The forecaster to wrap.
        cfg: Loaded configuration; ``conformal`` supplies alpha and the window.
    """

    def __init__(self, base: Forecaster, cfg: Config) -> None:
        self.base = base
        self.cfg = cfg
        self.residuals_: np.ndarray | None = None

    # -- Forecaster protocol -------------------------------------------------- #

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> ConformalForecaster:
        """Fit the wrapped model, then calibrate on the validation block."""
        self.base.fit(X, y, validation)
        if validation is not None:
            X_val, y_val = validation
            self.residuals_ = calibration_residuals(
                y_val,
                np.asarray(self.base.predict(X_val), dtype=float),
                minimum=self.cfg.conformal.min_calibration_residuals,
            )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Point forecasts, unchanged by the wrapper."""
        return np.asarray(self.base.predict(X), dtype=float)

    # -- intervals ------------------------------------------------------------ #

    def predict_interval(
        self, X: np.ndarray, alpha: float | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(lower, point, upper)`` for each row.

        Args:
            X: Feature tensor.
            alpha: Miss rate. ``0.2`` gives an 80% interval. Defaults to
                ``conformal.alpha``.

        Raises:
            UncertaintyError: no calibration residuals were collected, which means
                ``fit`` was called without a validation block.
        """
        if self.residuals_ is None or len(self.residuals_) == 0:
            raise UncertaintyError(
                "no calibration residuals: ConformalForecaster.fit needs a validation "
                "block. run_experiment supplies one from each fold."
            )
        alpha = self.cfg.conformal.alpha if alpha is None else alpha
        point = self.predict(X)
        width = conformal_width(self.residuals_, alpha)
        return point - width, point, point + width

    def predict_quantiles(
        self, X: np.ndarray, levels: np.ndarray
    ) -> np.ndarray:
        """Predicted quantiles at each level, shaped ``(n_levels, n_rows)``.

        Used to approximate CRPS, which needs more of the predictive distribution
        than a single interval exposes.
        """
        if self.residuals_ is None or len(self.residuals_) == 0:
            raise UncertaintyError("predict_quantiles called before calibration")
        point = self.predict(X)
        offsets = np.quantile(self.residuals_, levels)
        return point[None, :] + offsets[:, None]

    @property
    def calibration_size(self) -> int:
        """How many residuals the intervals rest on."""
        return 0 if self.residuals_ is None else int(len(self.residuals_))

    @property
    def diagnostics(self) -> dict[str, float]:
        """Pass through the wrapped model's diagnostics, plus calibration size."""
        inner = getattr(self.base, "diagnostics", None)
        out = dict(inner) if isinstance(inner, dict) else {}
        out["calibration_residuals"] = float(self.calibration_size)
        if self.residuals_ is not None and len(self.residuals_):
            out["conformal_width"] = float(
                conformal_width(self.residuals_, self.cfg.conformal.alpha)
            )
        return out

    @property
    def statistics(self) -> dict[str, list[float]] | None:
        """Pass through the wrapped scaler statistics, if the base has any."""
        inner = getattr(self.base, "statistics", None)
        return inner if isinstance(inner, dict) else None


# --------------------------------------------------------------------------- #
# The mechanics, as free functions so they can be tested without a model
# --------------------------------------------------------------------------- #


def calibration_residuals(
    actual: np.ndarray, predicted: np.ndarray, minimum: int
) -> np.ndarray:
    """Signed residuals over the whole calibration block.

    The rolling-window property comes from the fold, not from truncating here.
    :mod:`src.splits` gives each fold a validation block that is already a bounded,
    recent time window sitting just before the test period, and it slides forward
    as the folds advance. That is the rolling calibration window.

    Truncating to the last N *rows* would be actively wrong on a pooled panel.
    Rows are ordered state-major, so the last 24 rows are the last 24 periods of
    whichever state happens to sort last, not the last 24 periods across all
    states. The intervals would then be calibrated on one state and applied to
    twelve.

    Args:
        actual: Observed calibration values.
        predicted: Model predictions for the same rows.
        minimum: Fewest residuals that can support a quantile estimate.

    Raises:
        UncertaintyError: too few finite residuals to calibrate on.
    """
    residuals = np.asarray(actual, dtype=float) - np.asarray(predicted, dtype=float)
    residuals = residuals[np.isfinite(residuals)]
    if len(residuals) < minimum:
        raise UncertaintyError(
            f"only {len(residuals)} finite calibration residual(s), need at least "
            f"{minimum}; quantiles from fewer than that are noise. Widen the "
            "validation block via model.lstm.validation_fraction."
        )
    return residuals


def conformal_width(residuals: np.ndarray, alpha: float) -> float:
    """Half-width of a symmetric ``1 - alpha`` interval.

    The finite-sample correction ``ceil((n + 1)(1 - alpha)) / n`` is what gives
    split conformal its coverage guarantee. Without it a small calibration set
    produces intervals that are systematically slightly too narrow.
    """
    n = len(residuals)
    if n == 0:
        raise UncertaintyError("cannot compute a width from zero residuals")
    level = min(np.ceil((n + 1) * (1.0 - alpha)) / n, 1.0)
    return float(np.quantile(np.abs(residuals), level))


def predict_interval(
    model: Forecaster, X: np.ndarray, alpha: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(lower, point, upper)`` for a conformally-wrapped model.

    Raises:
        UncertaintyError: the model produces no intervals. Wrap it in
            :class:`ConformalForecaster` rather than reaching for a distributional
            assumption.
    """
    method = getattr(model, "predict_interval", None)
    if method is None:
        raise UncertaintyError(
            f"{type(model).__name__} does not produce intervals; wrap it with "
            "ConformalForecaster"
        )
    return method(X, alpha)


def coverage(actual: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Fraction of actuals falling inside the interval.

    The number to check against the nominal level. An 80% interval covering 55% is
    not conservative, it is wrong, and every alert built on its upper bound would
    fire late.
    """
    actual = np.asarray(actual, dtype=float)
    inside = (actual >= np.asarray(lower)) & (actual <= np.asarray(upper))
    return float(np.mean(inside))


def interval_width(lower: np.ndarray, upper: np.ndarray) -> float:
    """Mean interval width. Coverage alone is not enough.

    An interval from minus infinity to infinity covers everything and says
    nothing, so width is reported beside coverage.
    """
    return float(np.mean(np.asarray(upper) - np.asarray(lower)))


def crps_from_quantiles(
    actual: np.ndarray, quantile_predictions: np.ndarray, levels: np.ndarray
) -> float:
    """Approximate CRPS by averaging the pinball loss over quantile levels.

    CRPS scores the whole predictive distribution, not just the point forecast, so
    it is the metric that notices when intervals are well-centred but far too wide.
    The quantile-average form is the standard approximation when a model exposes
    quantiles rather than a density.

    Args:
        actual: ``(n_rows,)`` observed values.
        quantile_predictions: ``(n_levels, n_rows)`` predicted quantiles.
        levels: ``(n_levels,)`` quantile levels in ``(0, 1)``.
    """
    actual = np.asarray(actual, dtype=float)
    predictions = np.asarray(quantile_predictions, dtype=float)
    if predictions.shape != (len(levels), len(actual)):
        raise UncertaintyError(
            f"expected quantile predictions of shape {(len(levels), len(actual))}, "
            f"got {predictions.shape}"
        )
    error = actual[None, :] - predictions
    pinball = np.maximum(levels[:, None] * error, (levels[:, None] - 1.0) * error)
    return float(2.0 * pinball.mean())


def conformal(base_factory: ForecasterFactory, cfg: Config) -> ForecasterFactory:
    """Wrap a factory so every fold produces a calibrated interval model."""
    return lambda: ConformalForecaster(base_factory(), cfg)


_: type[Forecaster] = ConformalForecaster
