"""Splits, the evaluation harness, and the baselines — including the review gates."""

from __future__ import annotations

import dataclasses
import inspect
from itertools import pairwise

import numpy as np
import pandas as pd
import pytest

from src.config import Config
from src.evaluate import compare, compute_metrics, run_experiment
from src.features import TARGET_LEVEL_COLUMN, build_features
from src.models import Forecaster
from src.models.naive import BaselineError, LastValue, baseline_factories, persistence
from src.models.scaling import StandardScaled
from src.panel import complete_index
from src.splits import SplitError, rolling_origin


@pytest.fixture
def wide_cfg(cfg: Config) -> Config:
    """A window long enough for several folds on the synthetic panel."""
    return dataclasses.replace(
        cfg,
        data=dataclasses.replace(
            cfg.data,
            start_date=pd.Timestamp("2010-01-01").date(),
            end_date=pd.Timestamp("2019-12-31").date(),
        ),
        features=dataclasses.replace(cfg.features, sequence_length=3, lags=(1, 2, 12)),
        split=dataclasses.replace(
            cfg.split, n_folds=3, initial_train_size=60, test_size=12, step=12
        ),
    )


@pytest.fixture
def panel_wide(wide_cfg: Config) -> pd.DataFrame:
    """Seasonal synthetic panel with a mild trend, so models can differ."""
    index = complete_index(wide_cfg)
    month = index.get_level_values("date").month.to_numpy()
    step = np.arange(len(index)) % 120
    rng = np.random.default_rng(7)
    seasonal = np.sin(2 * np.pi * month / 12)
    return pd.DataFrame(
        {
            "cases": np.abs(60 + 40 * seasonal + 0.05 * step + rng.normal(0, 3, len(index))),
            "rainfall": 100 + 80 * seasonal + rng.normal(0, 5, len(index)),
            "temperature": 27 + 4 * seasonal + rng.normal(0, 0.5, len(index)),
            "humidity": 70 + 10 * seasonal,
            "search_interest": 40 + 20 * seasonal,
            "population": 3.0e7,
            "population_density": 850.0,
        },
        index=index,
    )


@pytest.fixture
def dataset(panel_wide: pd.DataFrame, wide_cfg: Config):
    """Built features for the wide panel."""
    return build_features(panel_wide, wide_cfg)


# --------------------------------------------------------------------------- #
# Splits: time-ordered, embargoed, never a state holdout
# --------------------------------------------------------------------------- #


def test_folds_split_on_time_not_on_states(dataset, wide_cfg: Config) -> None:
    """The central guarantee. A positional split would hold out states instead."""
    _, _, spec = dataset
    for fold in rolling_origin(spec.sample_index, wide_cfg, horizon=spec.horizon):
        train_states = set(spec.sample_index[fold.train].get_level_values("state"))
        test_states = set(spec.sample_index[fold.test].get_level_values("state"))
        assert train_states == test_states, "every state must appear on both sides"

        train_dates = spec.sample_index[fold.train].get_level_values("date")
        test_dates = spec.sample_index[fold.test].get_level_values("date")
        assert train_dates.max() < test_dates.min(), "training must end before testing starts"


def test_embargo_keeps_training_labels_out_of_the_test_window(
    dataset, wide_cfg: Config
) -> None:
    """A train sample at origin T carries a label from T+h; that must precede the test."""
    _, _, spec = dataset
    horizon = spec.horizon
    for fold in rolling_origin(spec.sample_index, wide_cfg, horizon=horizon):
        assert fold.embargo == horizon
        last_train_label = fold.val_end + pd.DateOffset(months=horizon)
        assert last_train_label < fold.test_start


def test_training_window_expands_across_folds(dataset, wide_cfg: Config) -> None:
    folds = list(rolling_origin(dataset[2].sample_index, wide_cfg))
    assert len(folds) == wide_cfg.split.n_folds
    sizes = [len(fold.fit) for fold in folds]
    assert sizes == sorted(sizes) and sizes[0] < sizes[-1]


def test_test_windows_do_not_overlap(dataset, wide_cfg: Config) -> None:
    folds = list(rolling_origin(dataset[2].sample_index, wide_cfg))
    for earlier, later in pairwise(folds):
        assert earlier.test_end < later.test_start


def test_val_is_the_tail_of_training_in_time_order(dataset, wide_cfg: Config) -> None:
    """A random validation carve would reintroduce the leakage splits exist to stop."""
    _, _, spec = dataset
    for fold in rolling_origin(spec.sample_index, wide_cfg):
        fit_dates = spec.sample_index[fold.fit].get_level_values("date")
        val_dates = spec.sample_index[fold.val].get_level_values("date")
        assert fit_dates.max() < val_dates.min()
        assert fold.fit_end < fold.val_end


def test_passing_a_sample_count_is_refused_with_an_explanation(wide_cfg: Config) -> None:
    """The signature change is load-bearing, so the old usage must not silently work."""
    with pytest.raises(SplitError, match="not a sample count"):
        list(rolling_origin(1836, wide_cfg))  # type: ignore[arg-type]


def test_geometry_that_does_not_fit_fails_loudly(dataset, wide_cfg: Config) -> None:
    """Silently yielding 2 folds while reporting 'mean across 5' would be false."""
    greedy = dataclasses.replace(wide_cfg, split=dataclasses.replace(wide_cfg.split, n_folds=20))
    with pytest.raises(SplitError, match="cannot support 20 folds"):
        list(rolling_origin(dataset[2].sample_index, greedy))


# --------------------------------------------------------------------------- #
# Review gate: the scaler is fitted inside the fold
# --------------------------------------------------------------------------- #


def test_scaler_statistics_differ_across_folds(dataset, wide_cfg: Config) -> None:
    """The gate, verified: a scaler fitted once outside would give identical stats."""
    X, y, spec = dataset
    means = []
    for fold in rolling_origin(spec.sample_index, wide_cfg, horizon=spec.horizon):
        scaled = StandardScaled(LastValue(0, "probe")).fit(X[fold.fit], y[fold.fit])
        means.append(np.asarray(scaled.statistics["mean"]))

    assert len(means) >= 2
    for earlier, later in pairwise(means):
        assert not np.allclose(earlier, later), "fold scalers must see different data"


def test_scaler_never_sees_the_test_window(dataset, wide_cfg: Config) -> None:
    """Fitting on fold.fit alone must give different statistics from fitting on everything."""
    X, y, spec = dataset
    fold = next(iter(rolling_origin(spec.sample_index, wide_cfg, horizon=spec.horizon)))

    inside = StandardScaled(LastValue(0, "probe")).fit(X[fold.fit], y[fold.fit])
    everything = StandardScaled(LastValue(0, "probe")).fit(X, y)
    assert not np.allclose(inside.statistics["mean"], everything.statistics["mean"])


def test_constant_features_do_not_divide_by_zero(dataset, wide_cfg: Config) -> None:
    X, y, spec = dataset
    scaled = StandardScaled(LastValue(0, "probe")).fit(X, y)
    assert np.isfinite(scaled.transform(X)).all()
    assert all(value > 0 for value in scaled.statistics["scale"])


# --------------------------------------------------------------------------- #
# Review gate: no branch on model type
# --------------------------------------------------------------------------- #


def test_run_experiment_never_branches_on_model_type() -> None:
    """If the harness needs isinstance, the Forecaster protocol is wrong."""
    import src.evaluate as module

    body = inspect.getsource(module).split('"""', 2)[-1]
    for forbidden in ("isinstance(model", "type(model)", "LSTM", "GBMBaseline", "LastValue"):
        assert forbidden not in body, f"{forbidden} appeared in src/evaluate.py"


def test_every_baseline_satisfies_the_protocol(dataset, wide_cfg: Config) -> None:
    _, _, spec = dataset
    for factory in baseline_factories(spec, wide_cfg).values():
        assert isinstance(factory(), Forecaster)


def test_the_readme_baseline_set_is_complete(dataset, wide_cfg: Config) -> None:
    """README section 6 asks for naive, linear and gradient-boosting baselines.

    Checked as a subset, not an equality: the set grows when a model earns its
    place, and pinning it exactly turns every addition into an unrelated test
    failure. What matters is that none of the four the README names goes missing.
    """
    _, _, spec = dataset
    assert {"persistence", "seasonal_naive", "gbm", "ridge"} <= set(
        baseline_factories(spec, wide_cfg)
    )


def test_ridge_baseline_learns_something(dataset, wide_cfg: Config) -> None:
    """The linear baseline must beat predicting the mean, or it is not a baseline."""
    from src.models.naive import ridge

    result = run_experiment(ridge(wide_cfg), dataset, wide_cfg, "ridge_probe")
    assert np.isfinite(result.primary)
    assert result.mean["r2_log"] > 0.0


def test_one_harness_serves_every_baseline_unmodified(dataset, wide_cfg: Config) -> None:
    """The gate: the same call runs all of them without special-casing."""
    _, _, spec = dataset
    factories = baseline_factories(spec, wide_cfg)
    results = [
        run_experiment(factory, dataset, wide_cfg, name)
        for name, factory in factories.items()
    ]
    assert len(results) == len(factories)
    for result in results:
        assert len(result.folds) == wide_cfg.split.n_folds
        assert np.isfinite(result.primary)


# --------------------------------------------------------------------------- #
# The harness
# --------------------------------------------------------------------------- #


def test_each_fold_gets_a_fresh_model(dataset, wide_cfg: Config) -> None:
    """A reused instance would accumulate state across folds without erroring."""
    seen: list[int] = []

    class _Counting(LastValue):
        def fit(self, X, y, validation=None):
            seen.append(id(self))
            return super().fit(X, y, validation)

    _, _, spec = dataset
    run_experiment(lambda: _Counting(0, "counting"), dataset, wide_cfg, "fresh")
    assert len(seen) == wide_cfg.split.n_folds
    assert len(set(seen)) == len(seen), "every fold must construct a new model"


def test_results_are_mean_and_std_across_folds_never_one_number(
    dataset, wide_cfg: Config
) -> None:
    _, _, spec = dataset
    result = run_experiment(persistence(spec), dataset, wide_cfg, "persistence")
    assert len(result.folds) == wide_cfg.split.n_folds
    assert set(result.mean) == set(result.std)
    assert result.std["mae_log"] >= 0.0
    assert "+/-" in result.summary()


def test_metrics_are_reported_on_both_scales(dataset, wide_cfg: Config) -> None:
    """Log-scale MAE is uninterpretable to a reader; raw-scale is uncomparable."""
    _, _, spec = dataset
    result = run_experiment(persistence(spec), dataset, wide_cfg, "persistence")
    for key in ("mae_log", "rmse_log", "r2_log",
                "mae_cases_per_100k", "rmse_cases_per_100k", "crps_cases_per_100k"):
        assert key in result.mean


def test_predictions_carry_state_and_target_date(dataset, wide_cfg: Config) -> None:
    """The dashboard maps predictions to states; the target date is not the origin."""
    _, _, spec = dataset
    result = run_experiment(persistence(spec), dataset, wide_cfg, "persistence")
    frame = result.predictions

    assert {"state", "origin_date", "target_date", "fold"} <= set(frame.columns)
    offset = (frame["target_date"] - frame["origin_date"]).dt.days
    assert (offset > 0).all(), "the target must lie after the origin"


def test_perfect_predictions_score_zero_error(wide_cfg: Config) -> None:
    actual = np.array([0.5, 1.0, 2.0, 3.0])
    metrics = compute_metrics(actual, actual.copy(), wide_cfg)
    assert metrics["mae_log"] == pytest.approx(0.0)
    assert metrics["rmse_log"] == pytest.approx(0.0)
    assert metrics["r2_log"] == pytest.approx(1.0)


def test_r2_is_undefined_not_zero_on_a_constant_window(wide_cfg: Config) -> None:
    """Reporting 0.0 would understate a forecast that is in fact exact."""
    actual = np.array([2.0, 2.0, 2.0])
    metrics = compute_metrics(actual, actual.copy(), wide_cfg)
    assert np.isnan(metrics["r2_log"])


def test_crps_equals_mae_for_a_point_forecast(wide_cfg: Config) -> None:
    """The degenerate case is exact, and is the bar Phase 6 intervals must beat."""
    actual = np.array([1.0, 2.0, 3.0])
    predicted = np.array([1.5, 1.0, 3.5])
    metrics = compute_metrics(actual, predicted, wide_cfg)
    assert metrics["crps_cases_per_100k"] == pytest.approx(metrics["mae_cases_per_100k"])


def test_non_finite_predictions_are_rejected(dataset, wide_cfg: Config) -> None:
    """NaNs would poison the fold averages silently."""
    from src.evaluate import EvaluationError

    class _Broken(LastValue):
        def predict(self, X):
            out = super().predict(X)
            out[0] = np.nan
            return out

    with pytest.raises(EvaluationError, match="non-finite"):
        run_experiment(lambda: _Broken(0, "broken"), dataset, wide_cfg, "broken")


# --------------------------------------------------------------------------- #
# Baselines
# --------------------------------------------------------------------------- #


def test_persistence_predicts_the_currently_observed_value(
    dataset, wide_cfg: Config
) -> None:
    X, _, spec = dataset
    model = persistence(spec)().fit(X[:10], np.zeros(10))
    column = spec.columns.index(f"{TARGET_LEVEL_COLUMN}_lag_0")
    np.testing.assert_allclose(model.predict(X[:10]), X[:10, -1, column])


def test_seasonal_naive_reads_the_period_before_the_target_not_the_origin(
    dataset, wide_cfg: Config
) -> None:
    """Same period last year relative to t+h is lag P-h, not lag P."""
    from src.models.naive import seasonal_naive

    _, _, spec = dataset
    expected = wide_cfg.project.seasonal_period - spec.horizon
    model = seasonal_naive(spec, wide_cfg)()
    assert spec.columns[model.column_index] == f"{TARGET_LEVEL_COLUMN}_lag_{expected}"


def test_baseline_without_its_column_explains_the_fix(
    panel_wide: pd.DataFrame, wide_cfg: Config
) -> None:
    no_target_lags = dataclasses.replace(
        wide_cfg, features=dataclasses.replace(wide_cfg.features, include_target_lags=False)
    )
    _, _, spec = build_features(panel_wide, no_target_lags)
    with pytest.raises(BaselineError, match="include_target_lags"):
        persistence(spec)


def test_compare_ranks_runs_by_the_headline_metric(dataset, wide_cfg: Config) -> None:
    _, _, spec = dataset
    results = [
        run_experiment(factory, dataset, wide_cfg, name)
        for name, factory in baseline_factories(spec, wide_cfg).items()
    ]
    table = compare(results)
    assert set(table.index) == {result.name for result in results}
    assert table["mae_cases_per_100k"].is_monotonic_increasing
