"""The experiment runner: one configuration in, metrics and artifacts out.

Every baseline, every ablation, every horizon and eventually the LSTM go through
:func:`run_experiment`. That is what keeps comparisons honest — the same folds,
the same embargo, the same metrics, computed by the same code for all of them.

Three properties this module is built around:

* **A factory, never an instance.** Each fold constructs a fresh model. Reusing one
  instance would continue training on the previous fold's fitted weights, which
  raises no error and silently contaminates the whole table.
* **No branching on model type.** Nothing here asks what kind of forecaster it
  holds. Models that need scaling are wrapped in
  :class:`~src.models.scaling.StandardScaled` by their factory; models that must
  see raw values are not. See that module for why the wrapper, rather than this
  runner, owns scaling.
* **Never a single-split number.** Results are reported as mean and standard
  deviation across folds. On a series this short, one split is close to a coin
  flip, and a single number invites reading noise as signal.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.artifacts import save_run
from src.config import Config
from src.features import FeatureSpec
from src.models import Forecaster
from src.splits import Fold, rolling_origin
from src.uncertainty import coverage, crps_from_quantiles, interval_width

#: Quantile levels used to approximate CRPS from a conformal model.
CRPS_LEVELS = np.round(np.arange(0.05, 1.0, 0.05), 2)

#: What :func:`run_experiment` consumes, exactly as ``build_features`` returns it.
Dataset = tuple[np.ndarray, np.ndarray, FeatureSpec]

#: Metric names computed on both the transformed and the back-transformed scale.
POINT_METRICS: tuple[str, ...] = ("mae", "rmse", "r2")


class EvaluationError(RuntimeError):
    """Raised when an experiment cannot be scored."""


@dataclass(frozen=True)
class FoldResult:
    """Everything one fold produced."""

    number: int
    n_fit: int
    n_val: int
    n_test: int
    metrics: dict[str, float]
    description: str
    scaler_statistics: dict[str, list[float]] | None = None
    diagnostics: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class RunResult:
    """A complete experiment: per-fold detail, aggregates, and predictions.

    Per-fold metrics are retained rather than only their average, because results
    are reported as mean +/- std and the spread is the interesting part.
    """

    name: str
    folds: tuple[FoldResult, ...]
    mean: dict[str, float]
    std: dict[str, float]
    predictions: pd.DataFrame
    config: dict[str, Any] = field(default_factory=dict)

    @property
    def primary(self) -> float:
        """Mean MAE on the interpretable scale — the number to quote."""
        return self.mean["mae_cases_per_100k"]

    def summary(self) -> str:
        """A readable block for the log and the report."""
        lines = [f"{self.name}  ({len(self.folds)} folds)"]
        for key in sorted(self.mean):
            lines.append(f"  {key:24s} {self.mean[key]:10.4f} +/- {self.std[key]:.4f}")
        return "\n".join(lines)

    def to_frame(self) -> pd.DataFrame:
        """One row per metric, for side-by-side comparison of several runs."""
        return pd.DataFrame(
            {"metric": list(self.mean), "mean": list(self.mean.values()),
             "std": [self.std[key] for key in self.mean]}
        ).assign(run=self.name)


def run_experiment(
    model_factory: Callable[[], Forecaster],
    data: Dataset,
    cfg: Config,
    name: str,
    *,
    save: bool = False,
) -> RunResult:
    """Fit and score one model across every rolling-origin fold.

    Args:
        model_factory: Zero-argument callable returning an unfitted forecaster. A
            factory rather than an instance so each fold trains from scratch.
        data: ``(X, y, spec)`` as returned by
            :func:`~src.features.build_features`.
        cfg: Loaded configuration.
        name: Run identifier, used for the artifact directory.
        save: Write predictions, metrics and the fold record to ``results/runs/``.

    Returns:
        The :class:`RunResult`, with metrics as mean +/- std across folds.

    Raises:
        EvaluationError: a fold has no test rows, or predictions are unusable.
    """
    X, y, spec = data
    fold_results: list[FoldResult] = []
    predictions: list[pd.DataFrame] = []

    for fold in rolling_origin(spec.sample_index, cfg, horizon=spec.horizon):
        _require_usable(fold, name)

        model = model_factory()
        # Handed to every model uniformly. Models that cannot use it ignore it,
        # which keeps this loop free of any branch on model type.
        model.fit(X[fold.fit], y[fold.fit], (X[fold.val], y[fold.val]))
        predicted = np.asarray(model.predict(X[fold.test]), dtype=float)
        _require_finite(predicted, fold, name)

        actual = y[fold.test]
        bounds = _interval(model, X[fold.test], cfg)
        metrics = compute_metrics(actual, predicted, cfg)
        metrics.update(_interval_metrics(model, X[fold.test], actual, bounds, cfg))

        fold_results.append(
            FoldResult(
                number=fold.number,
                n_fit=len(fold.fit),
                n_val=len(fold.val),
                n_test=len(fold.test),
                metrics=metrics,
                description=fold.describe(),
                scaler_statistics=_scaler_statistics(model),
                diagnostics=_model_diagnostics(model),
            )
        )
        predictions.append(_prediction_rows(fold, spec, actual, predicted, cfg, bounds))

    if not fold_results:
        raise EvaluationError(f"{name}: no folds were produced")

    mean, std = _aggregate(fold_results)
    result = RunResult(
        name=name,
        folds=tuple(fold_results),
        mean=mean,
        std=std,
        predictions=pd.concat(predictions, ignore_index=True),
        config=_run_config(cfg, spec, name),
    )

    if save:
        save_run(
            name,
            overwrite=True,
            predictions=result.predictions,
            metrics={"mean": mean, "std": std,
                     "folds": [_fold_record(item) for item in fold_results]},
            config=result.config,
        )
    return result


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def compute_metrics(
    actual: np.ndarray, predicted: np.ndarray, cfg: Config
) -> dict[str, float]:
    """Score one fold, on both the modelled and the interpretable scale.

    ``y`` is ``log(cases_per_100k + 1)``, which is the right scale to train on but
    means nothing to a public-health reader: an MAE of 0.3 log-units is not an
    answerable quantity. Both are therefore reported —  ``*_log`` for comparing
    models, ``*_cases_per_100k`` for the report.

    The back-transformation ``expm1`` is biased by Jensen's inequality: the mean of
    the exponential exceeds the exponential of the mean, so back-transformed errors
    read slightly high. That is a known and stated limitation, preferable to
    reporting only a figure nobody can interpret.

    CRPS is included for completeness. For a deterministic forecast the continuous
    ranked probability score reduces exactly to the absolute error, so this column
    is the bar any interval-producing model must beat once Phase 6 adds them.
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    metrics = {f"{key}_log": value for key, value in _point_metrics(actual, predicted).items()}

    if cfg.data.target_transform == "log1p":
        back_actual = np.expm1(actual)
        back_predicted = np.expm1(predicted)
        metrics.update(
            {
                f"{key}_cases_per_100k": value
                for key, value in _point_metrics(back_actual, back_predicted).items()
            }
        )
        metrics["crps_cases_per_100k"] = float(np.mean(np.abs(back_actual - back_predicted)))
    else:
        metrics.update({f"{key}_cases_per_100k": value for key, value in
                        _point_metrics(actual, predicted).items()})
        metrics["crps_cases_per_100k"] = metrics["mae_cases_per_100k"]

    return metrics


def _point_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """MAE, RMSE and R-squared for one pair of vectors."""
    error = actual - predicted
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error**2)))

    variance = float(np.sum((actual - actual.mean()) ** 2))
    # A constant test window has no variance to explain; R-squared is undefined
    # rather than zero, and reporting 0.0 would understate a perfect forecast.
    r2 = float("nan") if variance == 0 else 1.0 - float(np.sum(error**2)) / variance

    return {"mae": mae, "rmse": rmse, "r2": r2}


def _aggregate(
    folds: list[FoldResult],
) -> tuple[dict[str, float], dict[str, float]]:
    """Mean and standard deviation of each metric across folds."""
    keys = sorted(folds[0].metrics)
    mean: dict[str, float] = {}
    std: dict[str, float] = {}
    for key in keys:
        values = [fold.metrics[key] for fold in folds if np.isfinite(fold.metrics[key])]
        if not values:
            mean[key] = float("nan")
            std[key] = float("nan")
            continue
        mean[key] = float(statistics.fmean(values))
        std[key] = float(statistics.stdev(values)) if len(values) > 1 else 0.0
    return mean, std


# --------------------------------------------------------------------------- #
# Bookkeeping
# --------------------------------------------------------------------------- #


def _interval(
    model: Forecaster, X: np.ndarray, cfg: Config
) -> tuple[np.ndarray, np.ndarray] | None:
    """Interval bounds when the model produces them, otherwise None.

    Optional-protocol duck typing, the same pattern as diagnostics: a model either
    exposes ``predict_interval`` or it does not, and neither branch asks what type
    it is.
    """
    method = getattr(model, "predict_interval", None)
    if method is None:
        return None
    lower, _, upper = method(X, cfg.conformal.alpha)
    return np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)


def _interval_metrics(
    model: Forecaster,
    X: np.ndarray,
    actual: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray] | None,
    cfg: Config,
) -> dict[str, float]:
    """Coverage, width and a proper CRPS, when intervals exist.

    Coverage is the honesty check on the whole uncertainty story: an 80% interval
    should contain roughly 80% of held-out actuals. Width is reported beside it
    because an infinitely wide interval covers everything and says nothing.
    """
    if bounds is None:
        return {}
    lower, upper = bounds
    metrics = {
        "coverage": coverage(actual, lower, upper),
        "nominal_coverage": 1.0 - cfg.conformal.alpha,
        "interval_width_log": interval_width(lower, upper),
    }
    quantiles = getattr(model, "predict_quantiles", None)
    if quantiles is not None:
        metrics["crps_log"] = crps_from_quantiles(actual, quantiles(X, CRPS_LEVELS), CRPS_LEVELS)
    return metrics


def _prediction_rows(
    fold: Fold,
    spec: FeatureSpec,
    actual: np.ndarray,
    predicted: np.ndarray,
    cfg: Config,
    bounds: tuple[np.ndarray, np.ndarray] | None = None,
) -> pd.DataFrame:
    """Per-sample predictions, keyed so the dashboard can map them to states."""
    index = spec.sample_index[fold.test]
    frame = pd.DataFrame(
        {
            "state": index.get_level_values("state"),
            "origin_date": index.get_level_values("date"),
            "fold": fold.number,
            "actual_log": actual,
            "predicted_log": predicted,
        }
    )
    frame["target_date"] = frame["origin_date"] + _period_offset(cfg, spec.horizon)
    if bounds is not None:
        lower, upper = bounds
        frame["lower_log"] = lower
        frame["upper_log"] = upper
    if cfg.data.target_transform == "log1p":
        frame["actual_cases_per_100k"] = np.expm1(actual)
        frame["predicted_cases_per_100k"] = np.expm1(predicted)
        if bounds is not None:
            # The transform is monotone, so the back-transformed bounds are the
            # bounds of the back-transformed value. Clipped at zero: a negative
            # lower bound on a case count is not a statement about the world.
            frame["lower_cases_per_100k"] = np.maximum(np.expm1(bounds[0]), 0.0)
            frame["upper_cases_per_100k"] = np.maximum(np.expm1(bounds[1]), 0.0)
    return frame


def _period_offset(cfg: Config, horizon: int) -> pd.DateOffset:
    """Calendar distance from a forecast origin to the period it predicts."""
    if cfg.project.granularity == "monthly":
        return pd.DateOffset(months=horizon)
    return pd.DateOffset(weeks=horizon)


def _scaler_statistics(model: Forecaster) -> dict[str, list[float]] | None:
    """Fitted scaler statistics, when the model has a scaler to report.

    Optional duck-typing used only for the run record, never for control flow: a
    model without a scaler simply records nothing. The point is to make "the scaler
    was fitted inside this fold" a checkable claim.
    """
    statistics_attribute = getattr(model, "statistics", None)
    return statistics_attribute if isinstance(statistics_attribute, dict) else None


def _model_diagnostics(model: Forecaster) -> dict[str, float]:
    """Training diagnostics, when the model reports any.

    Optional duck-typing for the run record only, never for control flow. This is
    where the train/validation loss gap is captured, so "is the gap widening?" is
    answered from stored numbers rather than a console log.
    """
    reported = getattr(model, "diagnostics", None)
    return reported if isinstance(reported, dict) else {}


def _fold_record(fold: FoldResult) -> dict[str, Any]:
    """JSON-serialisable summary of one fold."""
    return {
        "fold": fold.number,
        "n_fit": fold.n_fit,
        "n_val": fold.n_val,
        "n_test": fold.n_test,
        "description": fold.description,
        "metrics": fold.metrics,
        "diagnostics": fold.diagnostics,
    }


def _run_config(cfg: Config, spec: FeatureSpec, name: str) -> dict[str, Any]:
    """The configuration that produced a run, stored beside its metrics.

    Every number in the report should be traceable to the exact settings behind it.
    """
    return {
        "run": name,
        "granularity": cfg.project.granularity,
        "horizon": spec.horizon,
        "target": spec.target_name,
        "timesteps": spec.timesteps,
        "n_features": spec.n_features,
        "feature_sources": list(cfg.features.sources),
        "include_lags": cfg.features.include_lags,
        "include_spatial": cfg.features.include_spatial,
        "include_target_lags": cfg.features.include_target_lags,
        "split": {
            "n_folds": cfg.split.n_folds,
            "initial_train_size": cfg.split.initial_train_size,
            "test_size": cfg.split.test_size,
            "step": cfg.split.step,
        },
    }


def _require_usable(fold: Fold, name: str) -> None:
    """Reject a fold that cannot be fitted or scored."""
    if len(fold.fit) == 0:
        raise EvaluationError(f"{name}: fold {fold.number} has no training rows")
    if len(fold.test) == 0:
        raise EvaluationError(f"{name}: fold {fold.number} has no test rows")


def _require_finite(predicted: np.ndarray, fold: Fold, name: str) -> None:
    """Reject NaN predictions rather than letting them poison the averages."""
    if not np.isfinite(predicted).all():
        count = int((~np.isfinite(predicted)).sum())
        raise EvaluationError(
            f"{name}: fold {fold.number} produced {count} non-finite prediction(s) "
            f"out of {len(predicted)}"
        )


def compare(results: list[RunResult]) -> pd.DataFrame:
    """Side-by-side table of several runs, best first on the headline metric."""
    frame = pd.concat([result.to_frame() for result in results], ignore_index=True)
    wide = frame.pivot(index="run", columns="metric", values="mean")
    return wide.sort_values("mae_cases_per_100k")
