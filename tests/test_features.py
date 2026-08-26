"""Feature engineering: correctness, purity, ablations, and the spec mapping."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from src.config import Config
from src.features import (
    CALENDAR_VARIABLE,
    STATE_VARIABLE,
    TRANSFORM_LAG,
    TRANSFORM_SPATIAL_LAG,
    FeatureError,
    add_lags,
    add_rolling,
    build_features,
    build_target,
    flatten,
)
from src.panel import complete_index


@pytest.fixture
def full_panel(cfg: Config) -> pd.DataFrame:
    """A panel carrying every variable the default config expects."""
    index = complete_index(cfg)
    month = index.get_level_values("date").month.to_numpy()
    rng = np.random.default_rng(0)
    seasonal = np.sin(2 * np.pi * month / 12)
    return pd.DataFrame(
        {
            "cases": 50 + 30 * seasonal + rng.normal(0, 2, len(index)),
            "rainfall": 100 + 80 * seasonal + rng.normal(0, 5, len(index)),
            "temperature": 27 + 4 * seasonal + rng.normal(0, 0.5, len(index)),
            "humidity": 70 + 10 * seasonal,
            "search_interest": 40 + 20 * seasonal,
            "population": 3.0e7,
            "population_density": 850.0,
        },
        index=index,
    )


def _shorten(cfg: Config, **feature_overrides: object) -> Config:
    """A config with a window short enough for a 36-month synthetic panel."""
    features = dataclasses.replace(
        cfg.features, sequence_length=3, lags=(1, 2, 12), **feature_overrides
    )
    return dataclasses.replace(cfg, features=features)


# --------------------------------------------------------------------------- #
# Review gate: purity
# --------------------------------------------------------------------------- #


def test_build_features_is_deterministic(cfg: Config, full_panel: pd.DataFrame) -> None:
    """Same input, same output — the simulator depends on it."""
    short = _shorten(cfg)
    first_X, first_y, first_spec = build_features(full_panel, short)
    second_X, second_y, second_spec = build_features(full_panel, short)

    np.testing.assert_array_equal(first_X, second_X)
    np.testing.assert_array_equal(first_y, second_y)
    assert first_spec.columns == second_spec.columns
    assert first_spec.sample_index.equals(second_spec.sample_index)


def test_build_features_does_not_mutate_the_panel(cfg: Config, full_panel: pd.DataFrame) -> None:
    """The simulator calls this on a panel it still needs afterwards."""
    before = full_panel.copy(deep=True)
    build_features(full_panel, _shorten(cfg))
    pd.testing.assert_frame_equal(full_panel, before)


def test_call_order_does_not_change_results(cfg: Config, full_panel: pd.DataFrame) -> None:
    """A hidden cache or accumulated state would show up here."""
    short = _shorten(cfg)
    modified = full_panel.copy()
    modified["rainfall"] *= 1.2

    baseline_first, _, _ = build_features(full_panel, short)
    _, _, _ = build_features(modified, short)
    baseline_again, _, _ = build_features(full_panel, short)

    np.testing.assert_array_equal(baseline_first, baseline_again)


def test_no_feature_sees_past_its_forecast_origin(
    cfg: Config, full_panel: pd.DataFrame
) -> None:
    """The leakage test: overwrite the future and every past sample must be identical.

    Stronger than inspecting shifts one at a time — it catches any path by which
    a later observation could reach an earlier sample, including through rolling
    windows and the spatial graph, without enumerating them.
    """
    short = _shorten(cfg)
    cut = pd.Timestamp("2017-01-01")

    baseline_X, baseline_y, spec = build_features(full_panel, short)
    tampered = full_panel.copy()
    tampered.loc[(slice(None), slice(cut, None)), :] = 1.0e6
    tampered_X, tampered_y, tampered_spec = build_features(tampered, short)

    assert spec.sample_index.equals(tampered_spec.sample_index)
    past = spec.sample_index.get_level_values("date") < cut
    assert past.sum() > 0

    np.testing.assert_allclose(baseline_X[past], tampered_X[past])
    # The target is supposed to look forward, so origins within one horizon of the
    # cut must move. If they did not, y would not be a forecast at all.
    within_horizon = (spec.sample_index.get_level_values("date") >= cut - pd.DateOffset(months=1)) & past
    assert not np.allclose(baseline_y[within_horizon], tampered_y[within_horizon])


def test_features_module_fits_nothing() -> None:
    """The review gate, enforced: no estimator may be constructed here."""
    import inspect

    import src.features as module

    body = inspect.getsource(module).split('"""', 2)[-1]
    for forbidden in (".fit(", ".fit_transform(", "Scaler(", "Encoder(", "lru_cache"):
        assert forbidden not in body, f"{forbidden} appeared in src/features.py"


# --------------------------------------------------------------------------- #
# Review gate: a new lagged variable is config alone
# --------------------------------------------------------------------------- #


def test_adding_a_lagged_variable_is_a_config_edit(cfg: Config, full_panel: pd.DataFrame) -> None:
    """No code change should be needed to lag a further driver."""
    without = _shorten(cfg, lag_variables=("cases", "rainfall"))
    with_extra = _shorten(cfg, lag_variables=("cases", "rainfall", "search_interest"))

    _, _, base_spec = build_features(full_panel, without)
    _, _, extended_spec = build_features(full_panel, with_extra)

    new = set(extended_spec.columns) - set(base_spec.columns)
    assert new == {f"search_interest_lag_{lag}" for lag in with_extra.features.lags}


def test_adding_a_lag_value_is_a_config_edit(cfg: Config, full_panel: pd.DataFrame) -> None:
    extended = _shorten(cfg)
    extended = dataclasses.replace(
        extended, features=dataclasses.replace(extended.features, lags=(1, 2, 3, 12))
    )
    _, _, spec = build_features(full_panel, extended)
    assert "rainfall_lag_3" in spec.columns


# --------------------------------------------------------------------------- #
# Review gate: FeatureSpec maps every column back to a raw variable
# --------------------------------------------------------------------------- #


def test_every_column_maps_back_to_a_raw_variable(cfg: Config, full_panel: pd.DataFrame) -> None:
    _, _, spec = build_features(full_panel, _shorten(cfg))
    for column in spec.columns:
        origin = spec.origins[column]
        assert origin.raw_variable in set(full_panel.columns) | {CALENDAR_VARIABLE, STATE_VARIABLE}


def test_flat_column_names_map_back_too(cfg: Config, full_panel: pd.DataFrame) -> None:
    """SHAP runs on the flat view, so 'rainfall_lag_1@t-2' must resolve."""
    X, _, spec = build_features(full_panel, _shorten(cfg))
    flat = flatten(X)

    assert flat.shape == (X.shape[0], spec.timesteps * spec.n_features)
    assert len(spec.flat_columns) == flat.shape[1]
    assert spec.raw_variable_of("rainfall_lag_1@t-2") == "rainfall"
    assert spec.raw_variable_of("rainfall_lag_1") == "rainfall"


def test_columns_from_finds_everything_the_simulator_must_move(
    cfg: Config, full_panel: pd.DataFrame
) -> None:
    """Raising rainfall must move its lags and rolling means together."""
    _, _, spec = build_features(full_panel, _shorten(cfg))
    derived = set(spec.columns_from("rainfall"))

    assert "rainfall" in derived
    assert {"rainfall_lag_1", "rainfall_lag_2", "rainfall_lag_12"} <= derived
    assert any(name.startswith("rainfall_roll") for name in derived)
    assert not any(name.startswith("temperature") for name in derived)


def test_attributions_group_back_to_raw_variables(cfg: Config, full_panel: pd.DataFrame) -> None:
    """A timesteps x features grid is unreadable; per-variable totals are not."""
    _, _, spec = build_features(full_panel, _shorten(cfg))
    values = np.ones(len(spec.flat_columns))
    totals = spec.group_by_raw_variable(values)

    assert sum(totals.values()) == pytest.approx(len(values))
    assert set(totals) <= set(full_panel.columns) | {CALENDAR_VARIABLE, STATE_VARIABLE}


def test_unknown_column_raises_rather_than_guessing(cfg: Config, full_panel: pd.DataFrame) -> None:
    _, _, spec = build_features(full_panel, _shorten(cfg))
    with pytest.raises(KeyError, match="unknown feature column"):
        spec.raw_variable_of("not_a_feature")


# --------------------------------------------------------------------------- #
# Transformation correctness
# --------------------------------------------------------------------------- #


def test_lags_never_cross_a_state_boundary(cfg: Config, full_panel: pd.DataFrame) -> None:
    """Kerala's first period must not borrow the previous state's last value."""
    built = add_lags(full_panel, ["cases"], [1])
    series, origin = built[0]
    assert origin.transform == TRANSFORM_LAG

    for state in full_panel.index.get_level_values("state").unique():
        first_date = full_panel.loc[state].index.min()
        assert np.isnan(series.loc[(state, first_date)])


def test_lag_values_are_the_previous_period(full_panel: pd.DataFrame) -> None:
    series, _ = add_lags(full_panel, ["cases"], [1])[0]
    kerala = full_panel.loc["Kerala", "cases"]
    assert series.loc[("Kerala", kerala.index[3])] == pytest.approx(kerala.iloc[2])


def test_rolling_windows_are_trailing_and_require_a_full_window(
    full_panel: pd.DataFrame,
) -> None:
    """A partial window would be a different statistic under the same name."""
    series, origin = add_rolling(full_panel, ["rainfall"], [3], ["mean"])[0]
    assert origin.window == 3

    kerala = full_panel.loc["Kerala", "rainfall"]
    assert np.isnan(series.loc[("Kerala", kerala.index[1])])
    assert series.loc[("Kerala", kerala.index[2])] == pytest.approx(kerala.iloc[:3].mean())


def test_target_is_the_future_rate_not_the_present(cfg: Config, full_panel: pd.DataFrame) -> None:
    target = build_target(full_panel, cfg, horizon=3)
    kerala_cases = full_panel.loc["Kerala", "cases"]
    dates = kerala_cases.index

    expected = np.log1p(
        kerala_cases.iloc[3] / 3.0e7 * cfg.data.population_normalisation
    )
    assert target.loc[("Kerala", dates[0])] == pytest.approx(expected)
    # The tail has no future to look at.
    assert np.isnan(target.loc[("Kerala", dates[-1])])


def test_target_does_not_shift_across_states(cfg: Config, full_panel: pd.DataFrame) -> None:
    target = build_target(full_panel, cfg, horizon=1)
    for state in full_panel.index.get_level_values("state").unique():
        last_date = full_panel.loc[state].index.max()
        assert np.isnan(target.loc[(state, last_date)])


def test_cyclic_features_are_continuous_across_the_year_boundary(
    cfg: Config, full_panel: pd.DataFrame
) -> None:
    """December and January must be adjacent, which a raw month index breaks."""
    _, _, spec = build_features(full_panel, _shorten(cfg))
    assert {"season_sin", "season_cos"} <= set(spec.columns)
    assert spec.origins["season_sin"].raw_variable == CALENDAR_VARIABLE


def test_spatial_lag_uses_neighbours_not_the_state_itself(
    cfg: Config, full_panel: pd.DataFrame
) -> None:
    """Kerala's spatial lag must come from Tamil Nadu, its only neighbour here."""
    short = _shorten(cfg, spatial_lags=(1,))
    _, _, spec = build_features(full_panel, short)
    column = "cases_spatial_lag_1"
    assert spec.origins[column].transform == TRANSFORM_SPATIAL_LAG

    spiked = full_panel.copy()
    spiked.loc[("Tamil Nadu", slice(None)), "cases"] = 9999.0
    _, _, _ = build_features(spiked, short)

    baseline_X, _, baseline_spec = build_features(full_panel, short)
    spiked_X, _, _ = build_features(spiked, short)
    position = baseline_spec.columns.index(column)

    kerala_rows = baseline_spec.sample_index.get_level_values("state") == "Kerala"
    assert not np.allclose(
        baseline_X[kerala_rows, :, position], spiked_X[kerala_rows, :, position]
    ), "Kerala's spatial lag should track its neighbour Tamil Nadu"

    odisha_rows = baseline_spec.sample_index.get_level_values("state") == "Odisha"
    np.testing.assert_allclose(
        baseline_X[odisha_rows, :, position], spiked_X[odisha_rows, :, position]
    ), "Odisha does not border Tamil Nadu"


# --------------------------------------------------------------------------- #
# Ablations are config, not code
# --------------------------------------------------------------------------- #


def test_climate_only_ablation_drops_case_inputs_but_keeps_the_target(
    cfg: Config, full_panel: pd.DataFrame
) -> None:
    """Configuration A must still have something to predict."""
    climate_only = dataclasses.replace(
        cfg,
        features=dataclasses.replace(
            _shorten(cfg).features, sources=("climate",), include_spatial=False
        ),
    )
    X, y, spec = build_features(full_panel, climate_only)

    assert "cases" not in spec.raw_variables
    assert "rainfall" in spec.raw_variables
    assert len(y) == X.shape[0] > 0
    assert np.isfinite(y).all()


def test_lag_ablation_removes_only_lag_and_rolling_columns(
    cfg: Config, full_panel: pd.DataFrame
) -> None:
    with_lags = _shorten(cfg)
    without = _shorten(cfg, include_lags=False)

    _, _, with_spec = build_features(full_panel, with_lags)
    _, _, without_spec = build_features(full_panel, without)

    removed = set(with_spec.columns) - set(without_spec.columns)
    assert removed
    assert all("_lag_" in name or "_roll" in name for name in removed)
    assert "rainfall" in without_spec.columns


def test_spatial_ablation_removes_only_spatial_columns(
    cfg: Config, full_panel: pd.DataFrame
) -> None:
    with_spatial = _shorten(cfg)
    without = _shorten(cfg, include_spatial=False)

    _, _, with_spec = build_features(full_panel, with_spatial)
    _, _, without_spec = build_features(full_panel, without)

    removed = set(with_spec.columns) - set(without_spec.columns)
    assert removed == {
        name for name in with_spec.columns if "_spatial_lag_" in name
    }


# --------------------------------------------------------------------------- #
# Shapes, indexing, and failure modes
# --------------------------------------------------------------------------- #


def test_shapes_and_index_line_up(cfg: Config, full_panel: pd.DataFrame) -> None:
    X, y, spec = build_features(full_panel, _shorten(cfg))
    assert X.ndim == 3
    assert X.shape[1] == spec.timesteps
    assert X.shape[2] == spec.n_features
    assert y.shape == (X.shape[0],)
    assert len(spec.sample_index) == X.shape[0]
    assert spec.sample_index.names == list(map(str, ["state", "date"]))


def test_sample_index_dates_are_the_forecast_origin(
    cfg: Config, full_panel: pd.DataFrame
) -> None:
    """The index marks the last observed period, not the period being predicted."""
    X, y, spec = build_features(full_panel, _shorten(cfg), horizon=2)
    state, origin_date = spec.sample_index[0]

    expected = build_target(full_panel, cfg, horizon=2).loc[(state, origin_date)]
    assert y[0] == pytest.approx(expected)


def test_windows_never_span_two_states(cfg: Config, full_panel: pd.DataFrame) -> None:
    """Every sample's window must lie wholly inside one state's series."""
    short = _shorten(cfg)
    _, _, spec = build_features(full_panel, short)
    per_state = spec.sample_index.get_level_values("state").value_counts()

    dates_per_state = len(full_panel.loc["Kerala"])
    # Each state contributes at most (periods - window + 1) windows.
    assert per_state.max() <= dates_per_state - short.features.sequence_length + 1


def test_samples_containing_a_gap_are_dropped_not_padded(
    cfg: Config, full_panel: pd.DataFrame
) -> None:
    holed = full_panel.copy()
    holed.loc[("Kerala", pd.Timestamp("2016-06-01")), "rainfall"] = np.nan

    _, _, baseline = build_features(full_panel, _shorten(cfg))
    X, _, spec = build_features(holed, _shorten(cfg))

    assert len(spec.sample_index) < len(baseline.sample_index)
    assert np.isfinite(X).all()


def test_impossible_window_raises_with_a_diagnostic(
    cfg: Config, full_panel: pd.DataFrame
) -> None:
    """A window longer than the series should say why, not return empty arrays."""
    impossible = dataclasses.replace(
        cfg, features=dataclasses.replace(cfg.features, sequence_length=500)
    )
    with pytest.raises(FeatureError, match="no complete samples"):
        build_features(full_panel, impossible)


def test_missing_target_columns_explain_the_ablation_distinction(
    cfg: Config, full_panel: pd.DataFrame
) -> None:
    with pytest.raises(FeatureError, match="features.sources controls model inputs"):
        build_target(full_panel.drop(columns=["population"]), cfg, horizon=1)


def test_unsorted_panel_is_rejected(cfg: Config, full_panel: pd.DataFrame) -> None:
    """Lags and windows assume sorted order; silently wrong output is worse."""
    from src.panel import PanelError

    shuffled = full_panel.sample(frac=1.0, random_state=0)
    with pytest.raises(PanelError, match="must be sorted"):
        build_features(shuffled, _shorten(cfg))

def test_a_state_with_no_in_study_neighbours_is_reported_not_silently_dropped(
    cfg: Config, full_panel: pd.DataFrame
) -> None:
    """Odisha borders none of Kerala or Tamil Nadu, so its spatial lag is all NaN.

    Every one of its windows is then incomplete and it leaves the dataset. Silent
    disappearance would also make the spatial ablation unfair, since configs E and
    F would be scored on different sets of states.
    """
    _, _, spec = build_features(full_panel, _shorten(cfg))
    assert "Odisha" in spec.dropped_states
    assert "Odisha" not in set(spec.sample_index.get_level_values("state"))

    without_spatial = _shorten(cfg, include_spatial=False)
    _, _, plain = build_features(full_panel, without_spatial)
    assert plain.dropped_states == ()
    assert "Odisha" in set(plain.sample_index.get_level_values("state"))
