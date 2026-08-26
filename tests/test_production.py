"""The frozen production artifact, and the boundary it draws around Phases 7-10."""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.artifacts import load_run, run_dir
from src.config import Config
from src.features import build_features
from src.panel import complete_index
from src.preprocess import preprocess
from src.production import (
    PRODUCTION_RUN,
    ProductionError,
    _time_ordered_tail_split,
    load_production,
    restore_config,
    select_configuration,
    train_production,
)

pytest.importorskip("keras", reason="production tests need keras")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: Modules that may only consume the frozen artifact.
DOWNSTREAM_MODULES = (
    "src/explain.py",
    "src/simulate.py",
    "src/recommend.py",
)

#: Names that would mean a downstream module is building or training its own model.
FORBIDDEN_NAMES = frozenset(
    {
        "PooledLSTM",
        "GBMBaseline",
        "LastValue",
        "ConformalForecaster",
        "StandardScaled",
        "pooled_lstm",
        "gradient_boosting",
        "persistence",
        "seasonal_naive",
        "baseline_factories",
        "baseline_builders",
        "run_experiment",
        "run_ablations",
        "train_production",
    }
)


@pytest.fixture
def tiny_cfg(cfg: Config) -> Config:
    """Small enough to fit in a couple of seconds."""
    from src.config import ExperimentSpec

    return dataclasses.replace(
        cfg,
        data=dataclasses.replace(
            cfg.data,
            start_date=pd.Timestamp("2012-01-01").date(),
            end_date=pd.Timestamp("2017-12-31").date(),
        ),
        features=dataclasses.replace(
            cfg.features, sequence_length=3, lags=(1, 2, 12), include_spatial=False
        ),
        forecast=dataclasses.replace(cfg.forecast, horizons=(1,)),
        model=dataclasses.replace(
            cfg.model,
            lstm=dataclasses.replace(
                cfg.model.lstm, units=8, layers=1, max_epochs=2, early_stopping_patience=1
            ),
        ),
        production=dataclasses.replace(cfg.production, calibration_periods=10),
        experiments=(
            ExperimentSpec(name="small", include_lags=False, include_spatial=False),
            ExperimentSpec(name="big", include_lags=True, include_spatial=False),
        ),
    )


@pytest.fixture
def tiny_panel(tiny_cfg: Config) -> pd.DataFrame:
    index = complete_index(tiny_cfg)
    month = index.get_level_values("date").month.to_numpy()
    rng = np.random.default_rng(5)
    seasonal = np.sin(2 * np.pi * month / 12)
    panel = pd.DataFrame(
        {
            "cases": np.abs(50 + 30 * seasonal + rng.normal(0, 3, len(index))),
            "rainfall": 100 + 80 * seasonal,
            "temperature": 27 + 4 * seasonal,
            "humidity": 70 + 10 * seasonal,
            "search_interest": 40 + 20 * seasonal,
            "population": 3.0e7,
            "population_density": 850.0,
        },
        index=index,
    )
    return preprocess(panel, tiny_cfg).panel[list(panel.columns)]


# --------------------------------------------------------------------------- #
# Review gate: the artifact is self-contained
# --------------------------------------------------------------------------- #


def test_artifact_carries_everything_needed_for_a_prediction(
    tiny_panel: pd.DataFrame, tiny_cfg: Config
) -> None:
    """The gate: model, scaler, feature spec, config, all in one run."""
    train_production(tiny_panel, tiny_cfg, experiment="small", horizon=1)

    payload = load_run(PRODUCTION_RUN)
    assert {"model", "scaler", "feature_spec", "config", "residuals", "trained_at"} <= set(
        payload
    )
    assert {"mean", "scale"} <= set(payload["scaler"])
    assert payload["feature_spec"]["columns"]
    assert len(payload["residuals"]) > 0


def test_raw_panel_to_prediction_in_one_call(
    tiny_panel: pd.DataFrame, tiny_cfg: Config
) -> None:
    """No other file, no retraining, no feature logic rebuilt by the caller."""
    artifact = train_production(tiny_panel, tiny_cfg, experiment="small", horizon=1)
    forecasts = artifact.predict(tiny_panel)

    assert {"state", "origin_date", "target_date", "predicted_log",
            "lower_log", "upper_log"} <= set(forecasts.columns)
    assert len(forecasts) > 0
    assert np.isfinite(forecasts["predicted_log"]).all()
    assert (forecasts["lower_log"] <= forecasts["predicted_log"]).all()
    assert (forecasts["predicted_log"] <= forecasts["upper_log"]).all()
    assert (forecasts["target_date"] > forecasts["origin_date"]).all()


def test_reloaded_artifact_reproduces_the_same_predictions(
    tiny_panel: pd.DataFrame, tiny_cfg: Config
) -> None:
    """A frozen model that changes on reload is not frozen."""
    trained = train_production(tiny_panel, tiny_cfg, experiment="small", horizon=1)
    before = trained.predict(tiny_panel)
    after = load_production().predict(tiny_panel)

    np.testing.assert_allclose(
        before["predicted_log"], after["predicted_log"], rtol=1e-5, atol=1e-6
    )


def test_loaded_config_comes_from_the_artifact_not_the_live_file(
    tiny_panel: pd.DataFrame, tiny_cfg: Config
) -> None:
    """The subtle one: the winning config may differ from what config.yaml says now.

    Reading the live file would build the wrong feature set for the frozen model.
    """
    train_production(tiny_panel, tiny_cfg, experiment="small", horizon=1)
    loaded = load_production()

    assert loaded.cfg.features.include_lags is False
    assert loaded.spec.n_features == len(loaded.spec.columns)


def test_restore_config_pins_features_but_not_paths(cfg: Config) -> None:
    """Paths describe this checkout; feature semantics describe the model."""
    import dataclasses

    features = dataclasses.replace(
        cfg.features, sources=("climate",), include_lags=False, sequence_length=4,
        lags=(1, 2, 12),
    )
    record = {
        "features": dataclasses.asdict(features),
        "data": {
            "target_column": "cases",
            "target_transform": "log1p",
            "population_normalisation": 100000,
        },
        "project": {"granularity": "monthly", "seed": 42},
        "conformal_alpha": 0.1,
    }
    restored = restore_config(record)
    assert restored.features.sources == ("climate",)
    assert restored.features.include_lags is False
    assert restored.features.sequence_length == 4
    # The field that was missed the first time: it changes the column set even
    # when include_lags is False, via the target's own autoregressive terms.
    assert restored.features.lags == (1, 2, 12)
    assert restored.conformal.alpha == 0.1
    # Paths come from the environment, not the artifact.
    assert restored.paths.runs.is_absolute()


def test_mismatched_panel_is_rejected_not_silently_predicted(
    tiny_panel: pd.DataFrame, tiny_cfg: Config
) -> None:
    """Silent column misalignment would produce confident nonsense."""
    artifact = train_production(tiny_panel, tiny_cfg, experiment="small", horizon=1)
    wrong = dataclasses.replace(
        artifact,
        cfg=dataclasses.replace(
            artifact.cfg,
            features=dataclasses.replace(artifact.cfg.features, include_lags=True),
        ),
    )
    with pytest.raises(ProductionError, match="does not produce the features"):
        wrong.predict(tiny_panel)


# --------------------------------------------------------------------------- #
# Review gate: re-running replaces the artifact everything else reads
# --------------------------------------------------------------------------- #


def test_refreezing_replaces_the_artifact_in_place(
    tiny_panel: pd.DataFrame, tiny_cfg: Config
) -> None:
    """The test of whether the boundary is real: one step, and downstream follows."""
    first = train_production(tiny_panel, tiny_cfg, experiment="small", horizon=1)
    assert load_production().experiment == "small"

    second = train_production(tiny_panel, tiny_cfg, experiment="big", horizon=1)
    reloaded = load_production()

    assert reloaded.experiment == "big"
    assert reloaded.spec.n_features == second.spec.n_features
    assert second.spec.n_features > first.spec.n_features
    assert reloaded.trained_at >= first.trained_at
    # One directory, replaced, not accumulated.
    assert sum(1 for _ in run_dir(PRODUCTION_RUN).glob("model.keras")) == 1


# --------------------------------------------------------------------------- #
# Fitting
# --------------------------------------------------------------------------- #


def test_calibration_is_held_out_by_period_not_by_row(
    tiny_panel: pd.DataFrame, tiny_cfg: Config
) -> None:
    """A row tail would hold out one state, not the most recent months."""
    _, _, spec = build_features(tiny_panel, tiny_cfg, horizon=1)
    fit_rows, calibration_rows = _time_ordered_tail_split(spec, calibration_periods=10)

    dates = pd.DatetimeIndex(spec.sample_index.get_level_values("date"))
    assert dates[fit_rows].max() < dates[calibration_rows].min()

    states = spec.sample_index.get_level_values("state")
    assert set(states[calibration_rows]) == set(states[fit_rows]), (
        "every state must appear in the calibration block"
    )


def test_holding_out_everything_is_refused(
    tiny_panel: pd.DataFrame, tiny_cfg: Config
) -> None:
    greedy = dataclasses.replace(
        tiny_cfg, production=dataclasses.replace(tiny_cfg.production, calibration_periods=999)
    )
    with pytest.raises(ProductionError, match="leaves"):
        train_production(tiny_panel, greedy, experiment="small", horizon=1, save=False)


def test_unknown_experiment_is_refused(tiny_panel: pd.DataFrame, tiny_cfg: Config) -> None:
    with pytest.raises(ProductionError, match="unknown experiment"):
        train_production(tiny_panel, tiny_cfg, experiment="nope", horizon=1, save=False)


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #


def test_selection_prefers_the_simplest_configuration_within_noise(cfg: Config) -> None:
    """A 72-feature win inside the noise is not a reason to ship 72 features."""
    table = pd.DataFrame(
        {
            "experiment": ["complex", "simple", "bad"],
            "model": ["lstm"] * 3,
            "horizon": [1, 1, 1],
            "n_features": [72, 29, 30],
            "mae_cases_per_100k": [0.0265, 0.0280, 0.0900],
            "mae_cases_per_100k_std": [0.0043, 0.0043, 0.0043],
        }
    )
    assert select_configuration(table, cfg) == ("simple", 1)


def test_selection_keeps_a_clear_winner(cfg: Config) -> None:
    """Parsimony must not override a difference that is actually distinguishable."""
    table = pd.DataFrame(
        {
            "experiment": ["complex", "simple"],
            "model": ["lstm"] * 2,
            "horizon": [1, 1],
            "n_features": [72, 29],
            "mae_cases_per_100k": [0.0200, 0.0900],
            "mae_cases_per_100k_std": [0.0010, 0.0010],
        }
    )
    assert select_configuration(table, cfg) == ("complex", 1)


def test_selection_restricts_to_the_configured_model(cfg: Config) -> None:
    """The primary model is fixed by the brief; a baseline winning goes in the report."""
    table = pd.DataFrame(
        {
            "experiment": ["a", "b"],
            "model": ["gbm", "lstm"],
            "horizon": [1, 1],
            "n_features": [29, 29],
            "mae_cases_per_100k": [0.0100, 0.0300],
            "mae_cases_per_100k_std": [0.0010, 0.0010],
        }
    )
    assert select_configuration(table, cfg)[0] == "b"


def test_selection_without_the_model_is_refused(cfg: Config) -> None:
    table = pd.DataFrame(
        {
            "experiment": ["a"], "model": ["gbm"], "horizon": [1],
            "n_features": [29], "mae_cases_per_100k": [0.01],
            "mae_cases_per_100k_std": [0.001],
        }
    )
    with pytest.raises(ProductionError, match="no ablation rows"):
        select_configuration(table, cfg)


# --------------------------------------------------------------------------- #
# Review gate: Phases 7-10 may only load the artifact
# --------------------------------------------------------------------------- #


def _module_paths() -> list[Path]:
    """Downstream modules plus anything under dashboard/."""
    paths = [PROJECT_ROOT / name for name in DOWNSTREAM_MODULES]
    paths.extend(sorted((PROJECT_ROOT / "dashboard").glob("**/*.py")))
    return [path for path in paths if path.is_file()]


@pytest.mark.parametrize("path", _module_paths(), ids=lambda p: p.name)
def test_downstream_modules_never_train(path: Path) -> None:
    """The boundary, enforced by parsing rather than grepping.

    A downstream module that builds its own model would explain, simulate or
    display a *different* model from the one making the forecast, and nothing would
    fail loudly when it happened.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            attribute = node.func.attr if isinstance(node.func, ast.Attribute) else None
            assert attribute not in {"fit", "fit_transform", "train_on_batch"}, (
                f"{path.name} calls .{attribute}(); it must load the frozen artifact"
            )

        if isinstance(node, ast.ImportFrom):
            imported = {alias.name for alias in node.names}
            clashes = imported & FORBIDDEN_NAMES
            assert not clashes, f"{path.name} imports model constructors {sorted(clashes)}"

        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            raise AssertionError(f"{path.name} references {node.id}")


def test_the_guard_would_catch_a_violation(tmp_path: Path) -> None:
    """A test that never fails is not a guard, so prove it fires."""
    offender = tmp_path / "offender.py"
    offender.write_text(
        "from src.models.lstm import PooledLSTM\n"
        "def go(spec, cfg, X, y):\n"
        "    return PooledLSTM(spec, cfg).fit(X, y)\n",
        encoding="utf-8",
    )
    with pytest.raises(AssertionError):
        test_downstream_modules_never_train(offender)
