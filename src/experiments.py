"""The ablation runner: one loop over configurations, not six scripts.

Every named configuration in ``config.yaml`` is applied to the base config,
features are rebuilt for it, and each model is put through the same
:func:`~src.evaluate.run_experiment`. Adding a configuration is an edit to the
YAML; adding a model is an entry in one dictionary. Neither needs a new file.

Two configurations deliberately produce fewer results than the others. The
climate-only ablation has no case history, so the naive baselines have nothing to
carry forward and cannot be built at all. That is recorded as a skip with its
reason rather than silently omitted, because a gap in a results table that nobody
explains reads like a failure.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from src.config import Config, ExperimentSpec
from src.evaluate import RunResult, run_experiment
from src.features import FeatureSpec, build_features
from src.models import ForecasterFactory
from src.models.lstm import pooled_lstm
from src.models.naive import BaselineError, baseline_builders
from src.uncertainty import conformal

#: Metrics carried into the tidy table, in reporting order.
REPORTED_METRICS: tuple[str, ...] = (
    "mae_cases_per_100k",
    "rmse_cases_per_100k",
    "mae_log",
    "rmse_log",
    "r2_log",
    "crps_log",
    "coverage",
    "interval_width_log",
)


class ExperimentError(RuntimeError):
    """Raised when an ablation grid cannot be run."""


@dataclass(frozen=True)
class Skipped:
    """A configuration and model pair that could not run, and why."""

    experiment: str
    model: str
    horizon: int
    reason: str


def apply_experiment(cfg: Config, spec: ExperimentSpec) -> Config:
    """Return the base config with one experiment's overrides applied.

    Only fields the spec actually sets are changed, so an entry states exactly
    what it varies and inherits everything else.
    """
    overrides = {
        field: getattr(spec, field)
        for field in ("sources", "include_lags", "include_spatial", "include_target_lags")
        if getattr(spec, field) is not None
    }
    if not overrides:
        return cfg
    return dataclasses.replace(
        cfg, features=dataclasses.replace(cfg.features, **overrides)
    )


def model_builders(
    spec: FeatureSpec, cfg: Config
) -> dict[str, Callable[[], ForecasterFactory]]:
    """Deferred constructors for every model in the grid.

    Baselines are point forecasters; the LSTM is wrapped in
    :class:`~src.uncertainty.ConformalForecaster` so the table carries coverage and
    CRPS for the model the project is actually about.
    """
    builders: dict[str, Callable[[], ForecasterFactory]] = dict(
        baseline_builders(spec, cfg)
    )
    builders["lstm"] = lambda: conformal(pooled_lstm(spec, cfg), cfg)
    return builders


def run_ablations(
    panel: pd.DataFrame,
    cfg: Config,
    *,
    models: tuple[str, ...] | None = None,
    save: bool = True,
    progress: Callable[[str], None] | None = None,
) -> tuple[pd.DataFrame, tuple[Skipped, ...]]:
    """Run every configuration in ``cfg.experiments`` at every configured horizon.

    Args:
        panel: Cleaned wide panel.
        cfg: Base configuration, carrying the experiment grid.
        models: Restrict to these model names. All of them by default.
        save: Write each run to ``results/runs/``.
        progress: Optional callback, called with a one-line status per run.

    Returns:
        A tidy frame with one row per (experiment, model, horizon) and mean and
        std for each reported metric, and the tuple of skipped combinations.
    """
    if not cfg.experiments:
        raise ExperimentError("config.yaml defines no experiments to run")

    rows: list[dict[str, object]] = []
    skipped: list[Skipped] = []

    for spec in cfg.experiments:
        variant = apply_experiment(cfg, spec)
        for horizon in cfg.forecast.horizons:
            try:
                data = build_features(panel, variant, horizon=horizon)
            except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
                skipped.append(Skipped(spec.name, "*", horizon, f"features: {exc}"))
                continue

            _, _, feature_spec = data
            builders = model_builders(feature_spec, variant)
            wanted = models or tuple(builders)

            for model_name in wanted:
                build = builders.get(model_name)
                if build is None:
                    skipped.append(
                        Skipped(spec.name, model_name, horizon, "no such model")
                    )
                    continue

                label = f"{spec.name}__{model_name}__h{horizon}"
                try:
                    factory = build()
                except BaselineError as exc:
                    skipped.append(
                        Skipped(spec.name, model_name, horizon, str(exc).splitlines()[0])
                    )
                    continue

                if progress:
                    progress(label)
                result = run_experiment(factory, data, variant, label, save=save)
                rows.append(_row(spec, model_name, horizon, feature_spec, result))

    if not rows:
        raise ExperimentError(
            f"every configuration failed; {len(skipped)} skip(s), first: "
            f"{skipped[0].reason if skipped else 'unknown'}"
        )
    return pd.DataFrame(rows), tuple(skipped)


def _row(
    spec: ExperimentSpec,
    model_name: str,
    horizon: int,
    feature_spec: FeatureSpec,
    result: RunResult,
) -> dict[str, object]:
    """One tidy row: what was run, and how it scored."""
    row: dict[str, object] = {
        "experiment": spec.name,
        "model": model_name,
        "horizon": horizon,
        "n_features": feature_spec.n_features,
        "n_samples": len(feature_spec.sample_index),
        "dropped_states": ", ".join(feature_spec.dropped_states),
        "folds": len(result.folds),
    }
    for metric in REPORTED_METRICS:
        row[metric] = result.mean.get(metric, float("nan"))
        row[f"{metric}_std"] = result.std.get(metric, float("nan"))
    return row


def significance(table: pd.DataFrame, metric: str = "mae_cases_per_100k") -> pd.DataFrame:
    """Compare each configuration against the best one on the given metric.

    Reports the gap in units of the pooled fold standard deviation. Anything under
    roughly one is not a distinguishable difference on this many folds, and the
    write-up should say so rather than declaring a winner.
    """
    if metric not in table.columns:
        raise ExperimentError(f"{metric!r} is not in the results table")

    out = []
    for (model, horizon), group in table.groupby(["model", "horizon"]):
        best = group.loc[group[metric].idxmin()]
        for _, row in group.iterrows():
            spread = float(max(row[f"{metric}_std"], best[f"{metric}_std"], 1e-12))
            out.append(
                {
                    "model": model,
                    "horizon": horizon,
                    "experiment": row["experiment"],
                    metric: row[metric],
                    "best_experiment": best["experiment"],
                    "gap": row[metric] - best[metric],
                    "gap_in_std": (row[metric] - best[metric]) / spread,
                    "distinguishable": bool(
                        (row[metric] - best[metric]) / spread >= 1.0
                    ),
                }
            )
    return pd.DataFrame(out).sort_values(["model", "horizon", metric])
