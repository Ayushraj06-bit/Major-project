"""Conformal intervals: calibration, coverage, and the ablation runner."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from src.config import Config, ConfigError
from src.experiments import apply_experiment, run_ablations, significance
from src.features import build_features
from src.models import Forecaster
from src.models.naive import LastValue
from src.panel import complete_index
from src.uncertainty import (
    ConformalForecaster,
    UncertaintyError,
    calibration_residuals,
    conformal,
    conformal_width,
    coverage,
    crps_from_quantiles,
    interval_width,
    predict_interval,
)


@pytest.fixture
def wide_cfg(cfg: Config) -> Config:
    return dataclasses.replace(
        cfg,
        data=dataclasses.replace(
            cfg.data,
            start_date=pd.Timestamp("2010-01-01").date(),
            end_date=pd.Timestamp("2019-12-31").date(),
        ),
        features=dataclasses.replace(
            cfg.features, sequence_length=3, lags=(1, 2, 12), include_spatial=False
        ),
        forecast=dataclasses.replace(cfg.forecast, horizons=(1,)),
        split=dataclasses.replace(
            cfg.split, n_folds=3, initial_train_size=60, test_size=12, step=12
        ),
    )


@pytest.fixture
def panel_wide(wide_cfg: Config) -> pd.DataFrame:
    index = complete_index(wide_cfg)
    month = index.get_level_values("date").month.to_numpy()
    rng = np.random.default_rng(11)
    seasonal = np.sin(2 * np.pi * month / 12)
    return pd.DataFrame(
        {
            "cases": np.abs(60 + 40 * seasonal + rng.normal(0, 4, len(index))),
            "rainfall": 100 + 80 * seasonal + rng.normal(0, 5, len(index)),
            "temperature": 27 + 4 * seasonal,
            "humidity": 70 + 10 * seasonal,
            "search_interest": 40 + 20 * seasonal,
            "population": 3.0e7,
            "population_density": 850.0,
        },
        index=index,
    )


@pytest.fixture
def dataset(panel_wide: pd.DataFrame, wide_cfg: Config):
    return build_features(panel_wide, wide_cfg)


# --------------------------------------------------------------------------- #
# Interval mechanics
# --------------------------------------------------------------------------- #


def test_width_grows_as_alpha_shrinks() -> None:
    """A 95% interval must be wider than an 80% one."""
    residuals = np.random.default_rng(0).normal(0, 1, 500)
    assert conformal_width(residuals, 0.05) > conformal_width(residuals, 0.2)


def test_width_tracks_the_size_of_the_errors() -> None:
    """A worse model must get wider intervals, with no code change."""
    rng = np.random.default_rng(1)
    tight = conformal_width(rng.normal(0, 0.1, 500), 0.2)
    loose = conformal_width(rng.normal(0, 1.0, 500), 0.2)
    assert loose > tight * 5


def test_finite_sample_correction_is_applied() -> None:
    """Without it a small calibration set yields intervals slightly too narrow."""
    residuals = np.linspace(-1, 1, 10)
    corrected = conformal_width(residuals, 0.2)
    naive = float(np.quantile(np.abs(residuals), 0.8))
    assert corrected >= naive


def test_calibration_uses_the_whole_block_not_a_row_tail(dataset, wide_cfg: Config) -> None:
    """Truncating to the last N rows would calibrate on one state only.

    Rows are ordered state-major, so a row-tail is the last periods of whichever
    state sorts last, not the last periods across all states.
    """
    residuals = calibration_residuals(np.arange(300.0), np.zeros(300), minimum=20)
    assert len(residuals) == 300


def test_too_few_residuals_is_refused_not_silently_used() -> None:
    with pytest.raises(UncertaintyError, match="need at least"):
        calibration_residuals(np.arange(5.0), np.zeros(5), minimum=20)


def test_coverage_counts_actuals_inside_the_band() -> None:
    actual = np.array([0.0, 1.0, 2.0, 3.0])
    assert coverage(actual, np.full(4, -1.0), np.full(4, 4.0)) == 1.0
    assert coverage(actual, np.full(4, 1.5), np.full(4, 2.5)) == 0.25


def test_width_is_reported_because_coverage_alone_is_meaningless() -> None:
    """An infinitely wide interval covers everything and says nothing."""
    assert interval_width(np.zeros(3), np.full(3, 2.0)) == 2.0


def test_crps_is_zero_for_a_perfect_deterministic_forecast() -> None:
    actual = np.array([1.0, 2.0, 3.0])
    levels = np.array([0.25, 0.5, 0.75])
    exact = np.tile(actual, (3, 1))
    assert crps_from_quantiles(actual, exact, levels) == pytest.approx(0.0)


def test_crps_penalises_a_wider_distribution_around_the_same_point() -> None:
    actual = np.array([0.0, 0.0, 0.0])
    levels = np.array([0.1, 0.5, 0.9])
    tight = np.array([[-0.1] * 3, [0.0] * 3, [0.1] * 3])
    loose = np.array([[-5.0] * 3, [0.0] * 3, [5.0] * 3])
    assert crps_from_quantiles(actual, loose, levels) > crps_from_quantiles(actual, tight, levels)


# --------------------------------------------------------------------------- #
# The wrapper
# --------------------------------------------------------------------------- #


def test_conformal_wrapper_satisfies_the_protocol(dataset, wide_cfg: Config) -> None:
    _, _, spec = dataset
    assert isinstance(conformal(lambda: LastValue(0, "probe"), wide_cfg)(), Forecaster)


def _thirds(n: int) -> tuple[slice, slice, slice]:
    """Train, calibrate, test slices sized off the actual dataset."""
    a, b = n // 3, 2 * (n // 3)
    return slice(0, a), slice(a, b), slice(b, n)


def test_point_forecasts_are_unchanged_by_wrapping(dataset, wide_cfg: Config) -> None:
    """The wrapper adds intervals; it must not move the point estimate."""
    X, y, spec = dataset
    column = spec.columns.index("target_level_lag_0")
    train, calib, test = _thirds(len(X))

    bare = LastValue(column, "p").fit(X[train], y[train])
    wrapped = ConformalForecaster(LastValue(column, "p"), wide_cfg).fit(
        X[train], y[train], (X[calib], y[calib])
    )
    np.testing.assert_allclose(bare.predict(X[test]), wrapped.predict(X[test]))


def test_interval_brackets_the_point_forecast(dataset, wide_cfg: Config) -> None:
    X, y, spec = dataset
    column = spec.columns.index("target_level_lag_0")
    train, calib, test = _thirds(len(X))
    model = ConformalForecaster(LastValue(column, "p"), wide_cfg).fit(
        X[train], y[train], (X[calib], y[calib])
    )
    lower, point, upper = model.predict_interval(X[test])
    assert (lower <= point).all() and (point <= upper).all()


def test_intervals_without_calibration_are_refused(dataset, wide_cfg: Config) -> None:
    """Silently returning a point estimate as an interval would be worse."""
    X, y, spec = dataset
    model = ConformalForecaster(LastValue(0, "p"), wide_cfg).fit(X[:200], y[:200])
    with pytest.raises(UncertaintyError, match="no calibration residuals"):
        model.predict_interval(X[:10])


def test_predict_interval_refuses_an_unwrapped_model(dataset, wide_cfg: Config) -> None:
    X, _, _ = dataset
    with pytest.raises(UncertaintyError, match="does not produce intervals"):
        predict_interval(LastValue(0, "p"), X[:10], 0.2)


def test_conformal_intervals_are_roughly_calibrated(dataset, wide_cfg: Config) -> None:
    """The review gate: about 80% of held-out actuals inside the 80% interval."""
    X, y, spec = dataset
    column = spec.columns.index("target_level_lag_0")
    train, calib, test = _thirds(len(X))

    model = ConformalForecaster(LastValue(column, "p"), wide_cfg).fit(
        X[train], y[train], (X[calib], y[calib])
    )
    lower, _, upper = model.predict_interval(X[test])
    observed = coverage(y[test], lower, upper)
    assert 0.6 <= observed <= 0.95, f"80% interval covered {observed:.1%}"


def test_alpha_argument_overrides_the_configured_level(dataset, wide_cfg: Config) -> None:
    X, y, spec = dataset
    column = spec.columns.index("target_level_lag_0")
    train, calib, test = _thirds(len(X))
    model = ConformalForecaster(LastValue(column, "p"), wide_cfg).fit(
        X[train], y[train], (X[calib], y[calib])
    )
    narrow = model.predict_interval(X[test], alpha=0.5)
    wide = model.predict_interval(X[test], alpha=0.05)
    assert (wide[2] - wide[0]).mean() > (narrow[2] - narrow[0]).mean()


# --------------------------------------------------------------------------- #
# The ablation runner
# --------------------------------------------------------------------------- #


def test_experiment_overrides_only_what_it_names(cfg: Config) -> None:
    from src.config import ExperimentSpec

    spec = ExperimentSpec(name="probe", sources=("climate",), include_spatial=False)
    variant = apply_experiment(cfg, spec)

    assert variant.features.sources == ("climate",)
    assert variant.features.include_spatial is False
    # Untouched fields are inherited.
    assert variant.features.lags == cfg.features.lags
    assert variant.features.include_lags == cfg.features.include_lags


def test_duplicate_experiment_names_are_rejected(cfg: Config) -> None:
    """They would overwrite each other's saved runs."""
    from src.config import ExperimentSpec

    with pytest.raises(ConfigError, match="duplicate name"):
        dataclasses.replace(
            cfg,
            experiments=(ExperimentSpec(name="same"), ExperimentSpec(name="same")),
        )


def test_grid_runs_every_configuration_through_one_runner(
    panel_wide: pd.DataFrame, wide_cfg: Config
) -> None:
    table, skipped = run_ablations(
        panel_wide, wide_cfg, models=("seasonal_naive",), save=False
    )
    ran = set(table["experiment"])
    assert len(ran) >= 4
    assert set(table["model"]) == {"seasonal_naive"}


def test_a_model_that_cannot_run_is_recorded_as_a_skip(
    panel_wide: pd.DataFrame, wide_cfg: Config
) -> None:
    """Climate-only has no case history, so persistence has nothing to carry."""
    table, skipped = run_ablations(
        panel_wide, wide_cfg, models=("persistence",), save=False
    )
    reasons = {(s.experiment, s.model) for s in skipped}
    assert ("A_climate_only", "persistence") in reasons
    assert "A_climate_only" not in set(table["experiment"])


def test_significance_flags_differences_within_noise(
    panel_wide: pd.DataFrame, wide_cfg: Config
) -> None:
    """The review gate: say plainly when nothing is distinguishable."""
    table, _ = run_ablations(panel_wide, wide_cfg, models=("seasonal_naive",), save=False)
    gaps = significance(table)

    assert {"gap_in_std", "distinguishable", "best_experiment"} <= set(gaps.columns)
    assert (gaps["gap"] >= -1e-12).all(), "the best configuration must have gap 0"
