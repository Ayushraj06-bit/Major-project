"""SHAP attribution: readable names, the flat wrapper, selection, and caching."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from src.config import Config
from src.explain import (
    Attribution,
    ExplainError,
    describe_column,
    explain,
    load_attribution,
    per_state_ranking,
    save_attribution,
    select_features,
)
from src.features import (
    CALENDAR_VARIABLE,
    STATE_VARIABLE,
    FeatureError,
    FeatureOrigin,
    build_features,
)
from src.panel import complete_index

pytest.importorskip("shap", reason="explainability tests need shap")
pytest.importorskip("keras", reason="explainability tests need keras")


@pytest.fixture
def tiny_cfg(cfg: Config) -> Config:
    """Small everything: SHAP is the slowest thing in the suite."""
    return dataclasses.replace(
        cfg,
        data=dataclasses.replace(
            cfg.data,
            start_date=pd.Timestamp("2014-01-01").date(),
            end_date=pd.Timestamp("2017-12-31").date(),
        ),
        features=dataclasses.replace(
            cfg.features, sequence_length=2, lags=(1, 12), rolling_windows=(3,),
            include_spatial=False,
        ),
        forecast=dataclasses.replace(cfg.forecast, horizons=(1,)),
        explain=dataclasses.replace(
            cfg.explain, background_samples=8, nsamples=32, max_explained_rows=6,
            top_k_features=4,
        ),
    )


@pytest.fixture
def tiny_data(tiny_cfg: Config):
    index = complete_index(tiny_cfg)
    month = index.get_level_values("date").month.to_numpy()
    rng = np.random.default_rng(9)
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
    return build_features(panel, tiny_cfg)


class _Linear:
    """A model whose attributions are known in advance, so SHAP can be checked."""

    def __init__(self, weights: np.ndarray) -> None:
        self.weights = weights

    def predict(self, X: np.ndarray, verbose: int = 0) -> np.ndarray:
        """Weighted sum over the flattened window."""
        return (X.reshape(len(X), -1) @ self.weights).reshape(-1, 1)


# --------------------------------------------------------------------------- #
# Review gate: attributions read in domain terms
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("origin", "expected"),
    [
        (FeatureOrigin("rainfall_lag_3", "rainfall", "lag", lag=3), "rainfall lag-3"),
        (FeatureOrigin("rainfall", "rainfall", "identity", lag=0), "rainfall (current)"),
        (
            FeatureOrigin("rainfall_roll3_mean", "rainfall", "rolling", window=3, agg="mean"),
            "rainfall 3-period mean",
        ),
        (
            FeatureOrigin("cases_spatial_lag_1", "cases", "spatial_lag", lag=1),
            "neighbouring cases lag-1",
        ),
        (FeatureOrigin("season_sin", CALENDAR_VARIABLE, "cyclic"), "calendar seasonality (sine)"),
        (
            FeatureOrigin("state_is_Kerala", STATE_VARIABLE, "static"),
            "state identity: Kerala",
        ),
        (
            FeatureOrigin("population_density", "population_density", "static"),
            "population density",
        ),
        (
            FeatureOrigin("search_interest_lag_2", "search_interest", "lag", lag=2),
            "search interest lag-2",
        ),
    ],
)
def test_columns_read_as_domain_phrases(origin: FeatureOrigin, expected: str) -> None:
    """The gate: 'rainfall lag-3', never 'feature_47'."""
    assert describe_column(origin) == expected


def test_target_lags_are_distinguished_from_raw_case_lags() -> None:
    """Both carry raw_variable 'cases' but they are different quantities.

    One is a count, the other the log case rate. Rendering both as 'cases lag-6'
    would leave a reader unable to tell which they are acting on.
    """
    count_lag = FeatureOrigin("cases_lag_6", "cases", "lag", lag=6)
    rate_lag = FeatureOrigin("target_level_lag_6", "cases", "lag", lag=6)

    assert describe_column(count_lag) == "cases lag-6"
    assert describe_column(rate_lag) == "case rate lag-6"
    assert describe_column(count_lag) != describe_column(rate_lag)


def test_every_production_column_gets_a_readable_name(tiny_data) -> None:
    """No column may fall through to its raw index."""
    _, _, spec = tiny_data
    for column in spec.columns:
        readable = describe_column(spec.origins[column])
        assert readable
        assert not readable.startswith("feature_")


# --------------------------------------------------------------------------- #
# The flat wrapper and attribution mechanics
# --------------------------------------------------------------------------- #


def test_explain_returns_one_value_per_feature_per_row(tiny_data, tiny_cfg: Config) -> None:
    """Summed over the time axis: a timesteps x features grid is unreadable."""
    X, _, spec = tiny_data
    model = _Linear(np.zeros(X.shape[1] * X.shape[2]))
    attribution = explain(model, X, spec, tiny_cfg)

    assert attribution.values.shape == (
        min(tiny_cfg.explain.max_explained_rows, len(X)),
        spec.n_features,
    )
    assert len(attribution.sample_index) == attribution.values.shape[0]


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_attributions_recover_a_known_linear_model(tiny_data, tiny_cfg: Config) -> None:
    """The weighted column must dominate, which is what checks the flat wrapper.

    If the ``reshape(-1, T, F)`` in the wrapper were transposed, attribution would
    land on the wrong column and this would fail.

    It asserts dominance rather than exclusivity on purpose. These synthetic
    features are near-collinear -- every one is driven by the same annual sine --
    and Shapley values split credit between correlated inputs by construction. A
    small amount of leakage onto neighbouring columns is correct behaviour, not a
    bug in the wrapper.
    """
    X, _, spec = tiny_data
    n_flat = X.shape[1] * X.shape[2]
    target_column = 1

    weights = np.zeros(n_flat)
    # Weight that column at every timestep.
    for step in range(X.shape[1]):
        weights[step * X.shape[2] + target_column] = 1.0

    attribution = explain(_Linear(weights), X, spec, tiny_cfg)
    magnitude = np.abs(attribution.values).mean(axis=0)
    others = np.delete(magnitude, target_column)

    assert int(np.argmax(magnitude)) == target_column
    assert magnitude[target_column] > 3.0 * others.max()


def test_attributions_are_reproducible(tiny_data, tiny_cfg: Config) -> None:
    """Same model, same data, same attributions.

    SHAP samples from NumPy's global generator, so without seeding these would
    depend on whatever last touched it -- which made this suite order-dependent
    and would make a cached dashboard explanation drift between rebuilds.
    """
    X, _, spec = tiny_data
    model = _Linear(np.ones(X.shape[1] * X.shape[2]))

    first = explain(model, X, spec, tiny_cfg)
    np.random.seed(12345)  # something else disturbs the global state
    second = explain(model, X, spec, tiny_cfg)

    np.testing.assert_allclose(first.values, second.values)
    assert first.base_value == pytest.approx(second.base_value)


def test_row_sample_spans_the_whole_panel(tiny_data, tiny_cfg: Config) -> None:
    """Evenly spaced, so the explanation is not one lucky season in one state."""
    X, _, spec = tiny_data
    attribution = explain(_Linear(np.zeros(X.shape[1] * X.shape[2])), X, spec, tiny_cfg)
    explained_states = set(attribution.sample_index.get_level_values("state"))
    assert len(explained_states) > 1


def test_unknown_explainer_is_rejected_by_config(cfg: Config) -> None:
    from src.config import ConfigError

    with pytest.raises(ConfigError, match="explain.explainer"):
        dataclasses.replace(cfg.explain, explainer="deep")


# --------------------------------------------------------------------------- #
# Aggregation views
# --------------------------------------------------------------------------- #


def test_importance_is_ranked_and_labelled(tiny_data, tiny_cfg: Config) -> None:
    X, _, spec = tiny_data
    attribution = explain(_Linear(np.ones(X.shape[1] * X.shape[2])), X, spec, tiny_cfg)
    importance = attribution.global_importance(spec)

    assert list(importance.columns) == ["column", "readable", "raw_variable", "mean_abs_shap"]
    assert importance["mean_abs_shap"].is_monotonic_decreasing


def test_by_raw_variable_sums_derived_columns_back(tiny_data, tiny_cfg: Config) -> None:
    """'Rainfall matters' is actionable; 'rainfall_roll3_mean at t-7' is not."""
    X, _, spec = tiny_data
    attribution = explain(_Linear(np.ones(X.shape[1] * X.shape[2])), X, spec, tiny_cfg)

    by_variable = attribution.by_raw_variable(spec)
    per_column = attribution.global_importance(spec)
    assert by_variable["mean_abs_shap"].sum() == pytest.approx(
        per_column["mean_abs_shap"].sum()
    )
    assert set(by_variable["raw_variable"]) <= set(spec.origins[c].raw_variable
                                                   for c in spec.columns)


def test_top_drivers_are_phrases_not_indices(tiny_data, tiny_cfg: Config) -> None:
    """This is what the recommendation layer quotes verbatim."""
    X, _, spec = tiny_data
    attribution = explain(_Linear(np.ones(X.shape[1] * X.shape[2])), X, spec, tiny_cfg)
    drivers = attribution.top_drivers(spec, row=0, k=3)

    assert len(drivers) == 3
    for label, value in drivers:
        assert isinstance(label, str) and " " in label
        assert np.isfinite(value)


def test_per_state_ranking_excludes_state_identity(tiny_data, tiny_cfg: Config) -> None:
    """A large attribution on 'state is Odisha' for a Kerala row is not a driver."""
    X, _, spec = tiny_data
    attribution = explain(_Linear(np.ones(X.shape[1] * X.shape[2])), X, spec, tiny_cfg)
    ranking = per_state_ranking(attribution, spec)

    columns = [column for column, _ in ranking.index]
    assert all(spec.origins[c].raw_variable != STATE_VARIABLE for c in columns)


# --------------------------------------------------------------------------- #
# Feature selection feeding back into the pipeline
# --------------------------------------------------------------------------- #


def test_selection_returns_top_k_drivers(tiny_data, tiny_cfg: Config) -> None:
    X, _, spec = tiny_data
    attribution = explain(_Linear(np.ones(X.shape[1] * X.shape[2])), X, spec, tiny_cfg)
    selected = select_features(attribution, spec, tiny_cfg)

    assert len(selected) == tiny_cfg.explain.top_k_features
    assert set(selected) <= set(spec.columns)
    assert all(spec.origins[c].raw_variable != STATE_VARIABLE for c in selected)


def test_selection_narrows_the_design_matrix_through_config(
    tiny_data, tiny_cfg: Config
) -> None:
    """Selection re-enters as configuration, not as a separate code path."""
    X, _, spec = tiny_data
    attribution = explain(_Linear(np.ones(X.shape[1] * X.shape[2])), X, spec, tiny_cfg)
    selected = select_features(attribution, spec, tiny_cfg)

    index = complete_index(tiny_cfg)
    month = index.get_level_values("date").month.to_numpy()
    seasonal = np.sin(2 * np.pi * month / 12)
    panel = pd.DataFrame(
        {
            "cases": np.abs(50 + 30 * seasonal), "rainfall": 100 + 80 * seasonal,
            "temperature": 27 + 4 * seasonal, "humidity": 70 + 10 * seasonal,
            "search_interest": 40 + 20 * seasonal, "population": 3.0e7,
            "population_density": 850.0,
        },
        index=index,
    )
    narrowed = dataclasses.replace(
        tiny_cfg,
        features=dataclasses.replace(tiny_cfg.features, selected_columns=selected),
    )
    _, _, narrow_spec = build_features(panel, narrowed)

    assert set(selected) <= set(narrow_spec.columns)
    assert narrow_spec.n_features < spec.n_features


def test_state_identity_survives_selection(tiny_data, tiny_cfg: Config) -> None:
    """Dropping the one-hots would change the architecture, not the feature set."""
    _, _, spec = tiny_data
    index = complete_index(tiny_cfg)
    month = index.get_level_values("date").month.to_numpy()
    seasonal = np.sin(2 * np.pi * month / 12)
    panel = pd.DataFrame(
        {
            "cases": np.abs(50 + 30 * seasonal), "rainfall": 100 + 80 * seasonal,
            "temperature": 27 + 4 * seasonal, "humidity": 70 + 10 * seasonal,
            "search_interest": 40 + 20 * seasonal, "population": 3.0e7,
            "population_density": 850.0,
        },
        index=index,
    )
    narrowed = dataclasses.replace(
        tiny_cfg,
        features=dataclasses.replace(tiny_cfg.features, selected_columns=("rainfall",)),
    )
    _, _, narrow_spec = build_features(panel, narrowed)
    assert len(narrow_spec.state_columns) > 0


def test_selection_naming_a_column_that_is_not_built_is_refused(
    tiny_cfg: Config,
) -> None:
    """A selection is only valid for the ablation flags it was produced under."""
    index = complete_index(tiny_cfg)
    panel = pd.DataFrame(
        {
            "cases": 50.0, "rainfall": 100.0, "temperature": 27.0, "humidity": 70.0,
            "search_interest": 40.0, "population": 3.0e7, "population_density": 850.0,
        },
        index=index,
    )
    bad = dataclasses.replace(
        tiny_cfg,
        features=dataclasses.replace(tiny_cfg.features, selected_columns=("not_a_column",)),
    )
    with pytest.raises(FeatureError, match="selected_columns"):
        build_features(panel, bad)


# --------------------------------------------------------------------------- #
# Caching
# --------------------------------------------------------------------------- #


def test_attributions_round_trip_through_the_artifact_store(
    tiny_data, tiny_cfg: Config
) -> None:
    """The dashboard must never recompute these; SHAP is minutes, not milliseconds."""
    X, _, spec = tiny_data
    attribution = explain(_Linear(np.ones(X.shape[1] * X.shape[2])), X, spec, tiny_cfg)
    save_attribution(attribution, spec, name="shap_probe")

    loaded = load_attribution("shap_probe")
    np.testing.assert_allclose(loaded.values, attribution.values)
    assert loaded.columns == attribution.columns
    assert loaded.sample_index.equals(attribution.sample_index)
    assert loaded.explainer == attribution.explainer


def test_missing_cache_says_how_to_produce_it() -> None:
    with pytest.raises(ExplainError, match="run scripts/run_shap.py"):
        load_attribution("never_computed")


def test_attribution_frame_is_labelled(tiny_data, tiny_cfg: Config) -> None:
    X, _, spec = tiny_data
    attribution = explain(_Linear(np.ones(X.shape[1] * X.shape[2])), X, spec, tiny_cfg)
    frame = attribution.frame()

    assert list(frame.columns) == list(spec.columns)
    assert frame.index.names == ["state", "date"]


def test_attribution_is_a_plain_dataclass() -> None:
    """Nothing in it should require a live model to interpret."""
    attribution = Attribution(
        values=np.zeros((2, 3)),
        base_value=0.0,
        columns=("a", "b", "c"),
        sample_index=pd.MultiIndex.from_tuples(
            [("Kerala", pd.Timestamp("2015-01-01")), ("Kerala", pd.Timestamp("2015-02-01"))],
            names=["state", "date"],
        ),
        explainer="kernel",
    )
    assert attribution.frame().shape == (2, 3)
