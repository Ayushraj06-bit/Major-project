"""The frozen production model: the one artifact everything downstream loads.

Every model trained so far has lived inside a cross-validation fold. Those exist to
produce a number and are thrown away; none of them is fit to serve a prediction,
because each saw only part of the data. This module produces the one that is.

The boundary this draws is the point of the module. Phases 7 to 10 — SHAP,
simulation, recommendation, dashboard — load :func:`load_production` and nothing
else. They never construct a model, never call ``fit``, and never read a fold. If
they did, the explanation on the dashboard would describe a different model from
the one making the forecast, and nobody would notice until someone checked.

**One deviation from "refit on the complete dataset", and it is not optional.**
The last ``production.calibration_periods`` periods are held out of the fit and used
to calibrate the conformal intervals. Calibrating on data the model has already
trained on produces intervals that are far too narrow, and the recommendation layer
alerts on the interval's upper bound. An overconfident upper bound means alerts
that fire late, which is the specific failure this project exists to avoid. The
model still sees every state and all but the final year of history.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from src.artifacts import load_run, run_dir, save_run
from src.config import Config, load_config
from src.experiments import apply_experiment
from src.features import FeatureSpec, build_features
from src.models.lstm import PooledLSTM, register_serializable_layers
from src.models.naive import gradient_boosting, persistence, seasonal_naive
from src.models.scaling import StandardScaled
from src.uncertainty import ConformalForecaster, conformal_width

#: The single run name every downstream phase loads.
PRODUCTION_RUN = "production"


class ProductionError(RuntimeError):
    """Raised when the production artifact cannot be built or loaded."""


@dataclass(frozen=True)
class ProductionModel:
    """A fitted model plus everything needed to use it, loaded from one artifact.

    Attributes:
        predictor: The fitted network or estimator.
        scaler_mean: Per-feature centring, fitted on the production training rows.
        scaler_scale: Per-feature scaling, fitted on the same rows.
        residuals: Conformal calibration residuals from the held-out tail.
        spec: Feature spec describing the columns the model expects.
        cfg: The exact configuration this model was built under.
        experiment: Which ablation configuration won.
        trained_at: When the artifact was produced.
    """

    predictor: Any
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    residuals: np.ndarray
    spec: FeatureSpec
    cfg: Config
    experiment: str
    trained_at: datetime

    def predict(self, panel: pd.DataFrame, *, alpha: float | None = None) -> pd.DataFrame:
        """Go from a raw panel to forecasts with intervals, in one call.

        This is the whole contract of the artifact: no other file, no retraining,
        no reconstruction of feature logic by the caller.

        Args:
            panel: A cleaned wide panel, same shape as training input.
            alpha: Miss rate for the interval. Defaults to ``conformal.alpha``.

        Returns:
            One row per ``(state, origin_date)`` with point forecast and interval,
            on both the modelled log scale and the interpretable case-rate scale.

        Raises:
            ProductionError: the panel produces features the model was not built
                for, which would otherwise be a silent misalignment.
        """
        X, _, spec = build_features(panel, self.cfg, horizon=self.spec.horizon)
        self._require_matching_columns(spec)

        scaled = (X - self.scaler_mean) / self.scaler_scale
        point = np.asarray(self.predictor.predict(scaled, verbose=0), dtype=float).ravel()

        alpha = self.cfg.conformal.alpha if alpha is None else alpha
        width = conformal_width(self.residuals, alpha)

        frame = pd.DataFrame(
            {
                "state": spec.sample_index.get_level_values("state"),
                "origin_date": spec.sample_index.get_level_values("date"),
                "predicted_log": point,
                "lower_log": point - width,
                "upper_log": point + width,
            }
        )
        frame["target_date"] = frame["origin_date"] + self._offset()
        if self.cfg.data.target_transform == "log1p":
            for column in ("predicted", "lower", "upper"):
                frame[f"{column}_cases_per_100k"] = np.maximum(
                    np.expm1(frame[f"{column}_log"]), 0.0
                )
        return frame

    def _offset(self) -> pd.DateOffset:
        """Calendar distance from a forecast origin to the period it predicts."""
        if self.cfg.project.granularity == "monthly":
            return pd.DateOffset(months=self.spec.horizon)
        return pd.DateOffset(weeks=self.spec.horizon)

    def _require_matching_columns(self, spec: FeatureSpec) -> None:
        """Reject a panel whose features do not line up with the frozen model."""
        if spec.columns != self.spec.columns:
            missing = sorted(set(self.spec.columns) - set(spec.columns))
            extra = sorted(set(spec.columns) - set(self.spec.columns))
            raise ProductionError(
                "this panel does not produce the features the production model was "
                f"fitted on. Missing {missing[:5]}, unexpected {extra[:5]}. The "
                "artifact carries its own config; do not override it."
            )
        if spec.timesteps != self.spec.timesteps:
            raise ProductionError(
                f"window mismatch: model expects {self.spec.timesteps} timesteps, "
                f"panel produced {spec.timesteps}"
            )


# --------------------------------------------------------------------------- #
# Selecting the winner
# --------------------------------------------------------------------------- #


def select_configuration(table: pd.DataFrame, cfg: Config) -> tuple[str, int]:
    """Pick the configuration and horizon to freeze, from the ablation table.

    Two rules, in order:

    1. Restrict to ``production.model``. The primary model is fixed by the project
       brief, so this chooses the best *configuration* for that model, not the best
       model overall. If a baseline beat it, that belongs in the write-up, not in
       the artifact.
    2. Among configurations within one fold standard deviation of the best, take
       the one with fewest features. A nominal win that is statistically
       indistinguishable is not a reason to ship three times the inputs, and a
       smaller model is cheaper to explain, simulate and serve.

    Returns:
        ``(experiment_name, horizon)``.

    Raises:
        ProductionError: the table has no rows for the configured model.
    """
    metric = cfg.production.selection_metric
    rows = table[table["model"] == cfg.production.model]
    if rows.empty:
        raise ProductionError(
            f"no ablation rows for production.model={cfg.production.model!r}; "
            f"table has {sorted(table['model'].unique())}"
        )
    if metric not in rows.columns:
        raise ProductionError(f"selection metric {metric!r} is not in the results table")

    best = rows.loc[rows[metric].idxmin()]
    if not cfg.production.parsimony_tiebreak:
        return str(best["experiment"]), int(best["horizon"])

    same_horizon = rows[rows["horizon"] == best["horizon"]]
    spread = float(max(best[f"{metric}_std"], 1e-12))
    tied = same_horizon[(same_horizon[metric] - best[metric]) / spread < 1.0]
    simplest = tied.loc[tied["n_features"].idxmin()]
    return str(simplest["experiment"]), int(best["horizon"])


# --------------------------------------------------------------------------- #
# Building the artifact
# --------------------------------------------------------------------------- #


def train_production(
    panel: pd.DataFrame,
    cfg: Config,
    *,
    experiment: str,
    horizon: int,
    save: bool = True,
) -> ProductionModel:
    """Fit the winning configuration once and write the production artifact.

    Args:
        panel: Cleaned wide panel, the complete dataset.
        cfg: Base configuration.
        experiment: Name of the winning ablation configuration.
        horizon: Forecast lead time to freeze.
        save: Write the artifact. Off only in tests.

    Returns:
        The fitted :class:`ProductionModel`.

    Raises:
        ProductionError: the named configuration is unknown, or too little data
            remains after holding out the calibration tail.
    """
    spec_lookup = {item.name: item for item in cfg.experiments}
    if experiment not in spec_lookup:
        raise ProductionError(
            f"unknown experiment {experiment!r}; config defines {sorted(spec_lookup)}"
        )

    variant = apply_experiment(cfg, spec_lookup[experiment])
    X, y, spec = build_features(panel, variant, horizon=horizon)

    fit_rows, calibration_rows = _time_ordered_tail_split(
        spec, cfg.production.calibration_periods
    )
    if len(fit_rows) == 0 or len(calibration_rows) == 0:
        raise ProductionError(
            f"holding out {cfg.production.calibration_periods} period(s) leaves "
            f"{len(fit_rows)} training and {len(calibration_rows)} calibration rows"
        )

    model = ConformalForecaster(StandardScaled(_build_model(variant, spec)), variant)
    model.fit(X[fit_rows], y[fit_rows], (X[calibration_rows], y[calibration_rows]))

    scaled: StandardScaled = model.base  # type: ignore[assignment]
    artifact = ProductionModel(
        predictor=_inner_predictor(scaled.base),
        scaler_mean=np.asarray(scaled.mean_, dtype=float),
        scaler_scale=np.asarray(scaled.scale_, dtype=float),
        residuals=np.asarray(model.residuals_, dtype=float),
        spec=spec,
        cfg=variant,
        experiment=experiment,
        trained_at=datetime.now(timezone.utc),
    )

    if save:
        _write(artifact, n_fit=len(fit_rows), n_calibration=len(calibration_rows))
    return artifact


def load_production() -> ProductionModel:
    """Load the frozen model. The only entry point Phases 7 to 10 may use.

    Raises:
        ProductionError: no artifact exists yet, or it is incomplete.
    """
    try:
        payload = load_run(PRODUCTION_RUN)
    except FileNotFoundError as exc:
        raise ProductionError(
            f"no production artifact at {run_dir(PRODUCTION_RUN)}. Run "
            "`python scripts/freeze_production.py` first; nothing downstream may "
            "train its own model."
        ) from exc

    required = {"model", "feature_spec", "scaler", "config", "trained_at", "residuals"}
    missing = sorted(required - set(payload))
    if missing:
        raise ProductionError(f"production artifact is missing {missing}")

    import keras

    # The custom column-routing layers register when this runs. Without it Keras
    # cannot resolve them and the load fails with an opaque deserialisation error.
    register_serializable_layers()

    scaler = payload["scaler"]
    meta = payload["trained_at"]
    return ProductionModel(
        predictor=keras.saving.load_model(payload["model"]),
        scaler_mean=np.asarray(scaler["mean"], dtype=float),
        scaler_scale=np.asarray(scaler["scale"], dtype=float),
        residuals=np.asarray(payload["residuals"], dtype=float),
        spec=FeatureSpec.from_dict(payload["feature_spec"]),
        cfg=restore_config(payload["config"]),
        experiment=str(meta["experiment"]),
        trained_at=datetime.fromisoformat(str(meta["timestamp"])),
    )


def restore_config(record: Mapping[str, Any]) -> Config:
    """Rebuild the feature configuration the artifact was trained under.

    Reading the live ``config.yaml`` instead would be a real bug, and a quiet one.
    The ablation that won may have had ``include_lags: false`` while the file on
    disk says true, in which case the panel would produce 70 features for a model
    fitted on 29. The column guard in :meth:`ProductionModel.predict` catches that,
    but only after somebody has already been confused by it.

    Machine-specific settings — paths above all — deliberately come from the live
    config, because they describe where this checkout keeps its data, not what the
    model was trained on.
    """
    import dataclasses

    from src.config import FeatureConfig

    base = load_config()
    stored: dict[str, Any] = {
        key: tuple(value) if isinstance(value, list) else value
        for key, value in dict(record["features"]).items()
    }
    features = FeatureConfig(**stored)
    return dataclasses.replace(
        base,
        features=features,
        data=dataclasses.replace(base.data, **dict(record["data"])),
        project=dataclasses.replace(base.project, **dict(record["project"])),
        conformal=dataclasses.replace(
            base.conformal, alpha=float(record["conformal_alpha"])
        ),
    )


def _write(artifact: ProductionModel, *, n_fit: int, n_calibration: int) -> None:
    """Persist every piece needed to reproduce a prediction from a raw panel."""
    save_run(
        PRODUCTION_RUN,
        overwrite=True,
        model=artifact.predictor,
        feature_spec=artifact.spec.to_dict(),
        scaler={
            "mean": artifact.scaler_mean.tolist(),
            "scale": artifact.scaler_scale.tolist(),
        },
        residuals=artifact.residuals,
        config=_config_record(artifact.cfg),
        trained_at={
            "timestamp": artifact.trained_at.isoformat(),
            "experiment": artifact.experiment,
            "horizon": artifact.spec.horizon,
            "n_fit_rows": n_fit,
            "n_calibration_rows": n_calibration,
            "n_features": artifact.spec.n_features,
        },
    )


def _config_record(cfg: Config) -> dict[str, object]:
    """The configuration that produced the artifact.

    Stores the **whole** feature config, not a hand-picked subset. Picking fields
    by hand is how this went wrong the first time: ``include_lags`` was pinned but
    ``lags`` was not, and the reloaded model was handed a different column set
    because the target's own autoregressive terms are derived from ``lags``
    regardless of the include flag. Anything that can change the column set has to
    travel with the artifact, and the safe way to guarantee that is to take all of
    it.
    """
    import dataclasses

    return {
        "features": dataclasses.asdict(cfg.features),
        "data": {
            "target_column": cfg.data.target_column,
            "target_transform": cfg.data.target_transform,
            "population_normalisation": cfg.data.population_normalisation,
        },
        "project": {
            "granularity": cfg.project.granularity,
            "seed": cfg.project.seed,
        },
        "conformal_alpha": cfg.conformal.alpha,
    }


def _time_ordered_tail_split(
    spec: FeatureSpec, calibration_periods: int
) -> tuple[np.ndarray, np.ndarray]:
    """Split rows by date, holding out the final periods for calibration.

    By period, not by row. Rows are ordered state-major, so a row-tail would hold
    out one state rather than the most recent months across all of them — the same
    trap that produced 72% coverage in Phase 6.
    """
    dates = pd.DatetimeIndex(spec.sample_index.get_level_values("date"))
    periods = pd.DatetimeIndex(sorted(pd.unique(dates)))
    if calibration_periods >= len(periods):
        return np.array([], dtype=int), np.arange(len(dates))

    cutoff = periods[-calibration_periods]
    return np.flatnonzero(dates < cutoff), np.flatnonzero(dates >= cutoff)


def _build_model(cfg: Config, spec: FeatureSpec) -> Any:
    """Construct the configured production model.

    The only place in the project outside the ablation runner that constructs a
    model. Phases 7 to 10 load the fitted artifact instead.
    """
    builders = {
        "lstm": lambda: PooledLSTM(spec, cfg),
        "gbm": lambda: gradient_boosting(cfg)(),
        "persistence": lambda: persistence(spec)(),
        "seasonal_naive": lambda: seasonal_naive(spec, cfg)(),
    }
    if cfg.production.model not in builders:
        raise ProductionError(
            f"production.model={cfg.production.model!r} is not buildable; "
            f"known models are {sorted(builders)}"
        )
    return builders[cfg.production.model]()


def _inner_predictor(model: Any) -> Any:
    """Unwrap to the object that actually holds the fitted weights.

    A Keras model is saved natively by the artifact store; anything else is
    pickled as-is.
    """
    return getattr(model, "model_", model)
