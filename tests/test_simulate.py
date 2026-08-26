"""Scenario simulation: coherence of the feature vector, and the guardrails."""

from __future__ import annotations

import ast
import dataclasses
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import Config
from src.features import build_features
from src.panel import complete_index
from src.simulate import (
    DIRECT,
    RECURSIVE,
    Scenario,
    SimulationError,
    apply_scenario,
    climatological_normals,
    forecast_horizon,
    plausible_range,
    simulate,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def sim_cfg(cfg: Config) -> Config:
    return dataclasses.replace(
        cfg,
        data=dataclasses.replace(
            cfg.data,
            start_date=pd.Timestamp("2014-01-01").date(),
            end_date=pd.Timestamp("2017-12-31").date(),
        ),
        features=dataclasses.replace(
            cfg.features, sequence_length=3, lags=(1, 2, 12), rolling_windows=(3,)
        ),
        forecast=dataclasses.replace(cfg.forecast, horizons=(1,)),
    )


@pytest.fixture
def sim_panel(sim_cfg: Config) -> pd.DataFrame:
    index = complete_index(sim_cfg)
    month = index.get_level_values("date").month.to_numpy()
    rng = np.random.default_rng(21)
    seasonal = np.sin(2 * np.pi * month / 12)
    panel = pd.DataFrame(
        {
            "cases": np.abs(50 + 30 * seasonal + rng.normal(0, 3, len(index))),
            "rainfall": np.abs(100 + 80 * seasonal + rng.normal(0, 10, len(index))),
            "temperature": 27 + 4 * seasonal + rng.normal(0, 1, len(index)),
            "humidity": 70 + 10 * seasonal,
            "search_interest": 40 + 20 * seasonal,
            "population": 3.0e7,
            "population_density": 850.0,
        },
        index=index,
    )
    # One genuine cloudburst per state, well outside +/-2 sigma. Real rainfall is
    # spiky, and without an outlier the clamp tests would never exercise the case
    # they exist for: a bound narrower than something that actually happened.
    for state in panel.index.get_level_values("state").unique():
        panel.loc[(state, pd.Timestamp("2015-08-01")), "rainfall"] = 900.0
    return panel


class _StubModel:
    """A stand-in production model: real features, deterministic prediction.

    Uses the genuine ``build_features`` so the coherence tests exercise the real
    pipeline, but sums the window instead of running a network, so the suite does
    not depend on a trained artifact.
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.calls = 0
        _, _, self.spec = build_features(_reference_panel(cfg), cfg)

    def predict(self, panel: pd.DataFrame) -> pd.DataFrame:
        self.calls += 1
        X, _, spec = build_features(panel, self.cfg, horizon=self.spec.horizon)
        point = np.nan_to_num(X).sum(axis=(1, 2)) / 1000.0
        frame = pd.DataFrame(
            {
                "state": spec.sample_index.get_level_values("state"),
                "origin_date": spec.sample_index.get_level_values("date"),
                "predicted_log": point,
                "lower_log": point - 0.1,
                "upper_log": point + 0.1,
            }
        )
        frame["target_date"] = frame["origin_date"] + pd.DateOffset(months=spec.horizon)
        # The same three rate columns the real ProductionModel emits. A double
        # that reports fewer columns than the thing it stands for lets a caller
        # pass here and fail in production.
        for column in ("predicted", "lower", "upper"):
            frame[f"{column}_cases_per_100k"] = np.maximum(
                np.expm1(frame[f"{column}_log"]), 0.0
            )
        return frame


def _reference_panel(cfg: Config) -> pd.DataFrame:
    index = complete_index(cfg)
    month = index.get_level_values("date").month.to_numpy()
    seasonal = np.sin(2 * np.pi * month / 12)
    return pd.DataFrame(
        {
            "cases": np.abs(50 + 30 * seasonal), "rainfall": np.abs(100 + 80 * seasonal),
            "temperature": 27 + 4 * seasonal, "humidity": 70 + 10 * seasonal,
            "search_interest": 40 + 20 * seasonal, "population": 3.0e7,
            "population_density": 850.0,
        },
        index=index,
    )


@pytest.fixture
def model(sim_cfg: Config) -> _StubModel:
    return _StubModel(sim_cfg)


# --------------------------------------------------------------------------- #
# Review gate: does simulation reuse build_features, or grow its own copy?
# --------------------------------------------------------------------------- #


def test_simulate_module_contains_no_feature_engineering() -> None:
    """The gate: a second copy of the lag logic is the failure this prevents.

    Parsed rather than grepped, so a comment mentioning lags does not trip it.
    """
    source = (PROJECT_ROOT / "src" / "simulate.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    defined = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    }
    forbidden = {"add_lags", "add_rolling", "add_spatial_lags", "build_features",
                 "add_cyclic", "window_sequences", "build_target"}
    assert not (defined & forbidden), f"simulate.py redefines {sorted(defined & forbidden)}"

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            attribute = node.func.attr if isinstance(node.func, ast.Attribute) else None
            assert attribute not in {"shift", "rolling"}, (
                f"simulate.py calls .{attribute}(); feature engineering belongs in "
                "src/features.py"
            )


def test_every_derived_column_moves_together(sim_cfg: Config, sim_panel: pd.DataFrame) -> None:
    """The property the gate exists to protect.

    Raising rainfall must move ``rainfall``, every ``rainfall_lag_k`` and every
    rolling mean. If only the contemporaneous column moved, the model would see a
    wet month whose own lag columns still describe a dry one.
    """
    scenario = Scenario(variable="rainfall", change=30.0)
    modified, _ = apply_scenario(sim_panel, scenario, sim_cfg)

    base_X, _, spec = build_features(sim_panel, sim_cfg)
    new_X, _, _ = build_features(modified, sim_cfg)

    derived = spec.columns_from("rainfall")
    assert len(derived) > 3, "expected the level, its lags and a rolling mean"

    for column in derived:
        position = spec.columns.index(column)
        assert not np.allclose(base_X[:, :, position], new_X[:, :, position]), (
            f"{column} did not move with rainfall"
        )


def test_columns_of_other_variables_do_not_move(
    sim_cfg: Config, sim_panel: pd.DataFrame
) -> None:
    """Coherence cuts both ways: temperature must be untouched by a rainfall change."""
    modified, _ = apply_scenario(
        sim_panel, Scenario(variable="rainfall", change=30.0), sim_cfg
    )
    base_X, _, spec = build_features(sim_panel, sim_cfg)
    new_X, _, _ = build_features(modified, sim_cfg)

    for column in spec.columns_from("temperature"):
        position = spec.columns.index(column)
        np.testing.assert_allclose(base_X[:, :, position], new_X[:, :, position])


# --------------------------------------------------------------------------- #
# Review gate: a null scenario reproduces the baseline exactly
# --------------------------------------------------------------------------- #


def test_zero_percent_scenario_reproduces_the_baseline_exactly(
    sim_cfg: Config, sim_panel: pd.DataFrame, model: _StubModel
) -> None:
    """The gate. Any difference here is a bug, and the likely cause is clamping."""
    result = simulate(sim_panel, Scenario(variable="rainfall", change=0.0), model, sim_cfg)

    np.testing.assert_array_equal(
        result.baseline["predicted_log"].to_numpy(),
        result.scenario_forecast["predicted_log"].to_numpy(),
    )
    assert result.delta["delta_log"].abs().max() == 0.0
    assert result.clamped_rows == 0
    assert result.out_of_distribution is False


def test_zero_absolute_scenario_also_reproduces_the_baseline(
    sim_cfg: Config, sim_panel: pd.DataFrame, model: _StubModel
) -> None:
    result = simulate(
        sim_panel, Scenario(variable="rainfall", change=0.0, mode="absolute"), model, sim_cfg
    )
    assert result.delta["delta_log"].abs().max() == 0.0


def test_a_sigma_clamp_does_not_rewrite_history(
    sim_cfg: Config, sim_panel: pd.DataFrame
) -> None:
    """A +/-2 sigma bound is narrower than the observed extremes.

    Without widening the bound to each cell's own value, a zero-change scenario
    would pull real outliers toward the mean and silently alter the panel.
    """
    assert sim_cfg.simulate.clamp_strategy == "sigma"
    bounds = plausible_range(sim_panel, "rainfall", sim_cfg)
    observed = sim_panel.groupby(level="state")["rainfall"].max()
    assert (observed > bounds["high"]).any(), "fixture must contain a genuine outlier"

    modified, clamped = apply_scenario(
        sim_panel, Scenario(variable="rainfall", change=0.0), sim_cfg
    )
    assert clamped == 0
    pd.testing.assert_frame_equal(modified, sim_panel)


# --------------------------------------------------------------------------- #
# Review gate: absurd inputs are clamped and flagged
# --------------------------------------------------------------------------- #


def test_absurd_scenario_is_clamped_and_flagged(
    sim_cfg: Config, sim_panel: pd.DataFrame, model: _StubModel
) -> None:
    """The gate: +2000% rainfall must not be answered silently."""
    result = simulate(
        sim_panel, Scenario(variable="rainfall", change=2000.0), model, sim_cfg
    )

    assert result.out_of_distribution is True
    assert result.clamped_rows > 0
    assert result.clamped_fraction > 0.9
    assert "OUT OF RANGE" in result.summary()


def test_clamped_values_stay_within_the_observed_range(
    sim_cfg: Config, sim_panel: pd.DataFrame
) -> None:
    modified, clamped = apply_scenario(
        sim_panel, Scenario(variable="rainfall", change=2000.0), sim_cfg
    )
    assert clamped > 0

    bounds = plausible_range(sim_panel, "rainfall", sim_cfg)
    for state in modified.index.get_level_values("state").unique():
        ceiling = max(
            bounds.loc[state, "high"], sim_panel.loc[state, "rainfall"].max()
        )
        assert modified.loc[state, "rainfall"].max() <= ceiling + 1e-9


def test_a_modest_scenario_barely_touches_the_guardrail(
    sim_cfg: Config, sim_panel: pd.DataFrame, model: _StubModel
) -> None:
    """A plausible question must not read as a wholesale extrapolation.

    +5% does clamp the record cloudburst cells, and correctly so: 5% above a
    record is still above the record. What separates it from an absurd scenario is
    the *share* of cells affected, which is why the fraction is reported alongside
    the flag.
    """
    modest = simulate(sim_panel, Scenario(variable="rainfall", change=5.0), model, sim_cfg)
    absurd = simulate(sim_panel, Scenario(variable="rainfall", change=2000.0), model, sim_cfg)

    assert modest.clamped_fraction < 0.05
    assert absurd.clamped_fraction > 0.9
    assert modest.delta["delta_log"].abs().max() > 0


# --------------------------------------------------------------------------- #
# Scenario semantics
# --------------------------------------------------------------------------- #


def test_scenario_can_target_specific_states(
    sim_cfg: Config, sim_panel: pd.DataFrame
) -> None:
    modified, _ = apply_scenario(
        sim_panel, Scenario(variable="rainfall", change=20.0, states=("Kerala",)), sim_cfg
    )
    assert not np.allclose(
        modified.loc["Kerala", "rainfall"], sim_panel.loc["Kerala", "rainfall"]
    )
    np.testing.assert_allclose(
        modified.loc["Odisha", "rainfall"], sim_panel.loc["Odisha", "rainfall"]
    )


def test_scenario_can_target_a_time_span(sim_cfg: Config, sim_panel: pd.DataFrame) -> None:
    scenario = Scenario(
        variable="rainfall", change=50.0,
        start=date(2016, 1, 1), end=date(2016, 6, 30),
    )
    modified, _ = apply_scenario(sim_panel, scenario, sim_cfg)
    dates = sim_panel.index.get_level_values("date")

    inside = (dates >= pd.Timestamp("2016-01-01")) & (dates <= pd.Timestamp("2016-06-30"))
    assert not np.allclose(modified["rainfall"][inside], sim_panel["rainfall"][inside])
    np.testing.assert_allclose(
        modified["rainfall"][~inside], sim_panel["rainfall"][~inside]
    )


def test_absolute_and_percent_modes_differ(sim_cfg: Config, sim_panel: pd.DataFrame) -> None:
    percent, _ = apply_scenario(
        sim_panel, Scenario(variable="temperature", change=10.0, mode="percent"), sim_cfg
    )
    absolute, _ = apply_scenario(
        sim_panel, Scenario(variable="temperature", change=10.0, mode="absolute"), sim_cfg
    )
    assert not np.allclose(percent["temperature"], absolute["temperature"])


def test_panel_is_never_mutated(
    sim_cfg: Config, sim_panel: pd.DataFrame, model: _StubModel
) -> None:
    """The caller still needs the original; a scenario must not consume it."""
    before = sim_panel.copy(deep=True)
    simulate(sim_panel, Scenario(variable="rainfall", change=40.0), model, sim_cfg)
    pd.testing.assert_frame_equal(sim_panel, before)


def test_unknown_variable_is_refused(
    sim_cfg: Config, sim_panel: pd.DataFrame, model: _StubModel
) -> None:
    with pytest.raises(SimulationError, match="not a panel column"):
        simulate(sim_panel, Scenario(variable="sunshine", change=10.0), model, sim_cfg)


def test_scenario_selecting_nothing_is_refused(
    sim_cfg: Config, sim_panel: pd.DataFrame, model: _StubModel
) -> None:
    """Silently returning a zero delta would read as 'rainfall does not matter'."""
    scenario = Scenario(
        variable="rainfall", change=10.0,
        start=date(2030, 1, 1), end=date(2031, 1, 1),
    )
    with pytest.raises(SimulationError, match="selects no rows"):
        simulate(sim_panel, scenario, model, sim_cfg)


def test_invalid_mode_is_refused() -> None:
    with pytest.raises(SimulationError, match="Scenario.mode"):
        Scenario(variable="rainfall", change=10.0, mode="multiply")


def test_reversed_time_span_is_refused() -> None:
    with pytest.raises(SimulationError, match="is after end"):
        Scenario(
            variable="rainfall", change=10.0,
            start=date(2017, 1, 1), end=date(2016, 1, 1),
        )


# --------------------------------------------------------------------------- #
# Result reporting
# --------------------------------------------------------------------------- #


def test_result_carries_both_forecasts_intervals_and_delta(
    sim_cfg: Config, sim_panel: pd.DataFrame, model: _StubModel
) -> None:
    result = simulate(sim_panel, Scenario(variable="rainfall", change=25.0), model, sim_cfg)

    assert {"lower_base", "upper_base", "lower_scenario", "upper_scenario"} <= set(
        result.delta.columns
    )
    assert len(result.baseline) == len(result.scenario_forecast) == len(result.delta)
    assert np.isfinite(result.mean_delta)


def test_summary_always_states_the_causation_caveat(
    sim_cfg: Config, sim_panel: pd.DataFrame, model: _StubModel
) -> None:
    """It has to travel with the number, not sit in a docstring nobody reads."""
    result = simulate(sim_panel, Scenario(variable="rainfall", change=15.0), model, sim_cfg)
    assert "correlation, not causation" in result.summary()


def test_a_variable_the_model_ignores_is_reported_as_such(
    sim_cfg: Config, sim_panel: pd.DataFrame
) -> None:
    """A zero delta is ambiguous unless you know whether the model uses the input."""
    climate_only = dataclasses.replace(
        sim_cfg,
        features=dataclasses.replace(
            sim_cfg.features, sources=("climate",), include_spatial=False,
            include_target_lags=False,
        ),
    )
    model = _StubModel(climate_only)
    result = simulate(
        sim_panel, Scenario(variable="search_interest", change=50.0), model, climate_only
    )

    assert result.affects_model is False
    assert "feeds no model input" in result.summary()


def test_scenario_description_is_readable() -> None:
    scenario = Scenario(variable="rainfall", change=20.0, states=("Kerala",))
    assert scenario.describe() == "+20% rainfall in Kerala"
    assert Scenario(variable="rainfall", change=0.0).is_null


# --------------------------------------------------------------------------- #
# Forward projection
# --------------------------------------------------------------------------- #


def _project(
    panel: pd.DataFrame, cfg: Config, model: _StubModel, months: int
) -> object:
    """Project Kerala ``months`` past its last observation."""
    last = panel.loc["Kerala", "cases"].dropna().index.max()
    return forecast_horizon(
        panel, "Kerala", pd.Timestamp(last) + pd.DateOffset(months=months), model, cfg
    )


def test_within_horizon_is_direct_and_fully_reliable(
    sim_panel: pd.DataFrame, sim_cfg: Config, model: _StubModel
) -> None:
    """A one-step-ahead forecast used only observed inputs, so it is not degraded."""
    curve = _project(sim_panel, sim_cfg, model, months=1)

    assert [step.mode for step in curve.steps] == [DIRECT]
    assert curve.reliability == 1.0
    assert not curve.has_recursive
    assert not curve.truncated


def test_beyond_horizon_is_labelled_recursive(
    sim_panel: pd.DataFrame, sim_cfg: Config, model: _StubModel
) -> None:
    """The boundary has to survive to the caller.

    A five-month projection and a one-month forecast are different kinds of claim,
    and a UI that cannot tell them apart will present them identically.
    """
    curve = _project(sim_panel, sim_cfg, model, months=4)

    modes = [step.mode for step in curve.steps]
    assert modes[0] == DIRECT
    assert set(modes[1:]) == {RECURSIVE}
    assert curve.has_recursive
    assert curve.direct_until == curve.last_observed + pd.DateOffset(months=1)


def test_curve_covers_every_period_without_gaps(
    sim_panel: pd.DataFrame, sim_cfg: Config, model: _StubModel
) -> None:
    """The path, not just the endpoint -- and every step counted from one origin."""
    curve = _project(sim_panel, sim_cfg, model, months=4)

    assert [step.steps_ahead for step in curve.steps] == [1, 2, 3, 4]
    expected = [
        curve.last_observed + pd.DateOffset(months=step) for step in range(1, 5)
    ]
    assert [step.target_date for step in curve.steps] == expected


def test_interval_widens_with_every_recursive_step(
    sim_panel: pd.DataFrame, sim_cfg: Config, model: _StubModel
) -> None:
    """A flat band four months out would imply confidence the method cannot support."""
    curve = _project(sim_panel, sim_cfg, model, months=5)

    widths = [
        step.upper_cases_per_100k - step.lower_cases_per_100k for step in curve.steps
    ]
    assert widths == sorted(widths)
    assert widths[-1] > widths[0]


def test_reliability_decays_with_depth(
    sim_panel: pd.DataFrame, sim_cfg: Config, model: _StubModel
) -> None:
    curve = _project(sim_panel, sim_cfg, model, months=5)

    scores = [step.reliability for step in curve.steps]
    assert scores == sorted(scores, reverse=True)
    assert curve.reliability == scores[-1]


def test_request_past_the_cap_is_truncated_not_answered(
    sim_panel: pd.DataFrame, sim_cfg: Config, model: _StubModel
) -> None:
    """Refusing loudly beats extrapolating quietly."""
    curve = _project(sim_panel, sim_cfg, model, months=48)

    assert curve.truncated
    assert len(curve.steps) <= 1 + sim_cfg.forecast.max_recursive_steps
    assert curve.steps[-1].target_date <= curve.direct_until + pd.DateOffset(
        months=sim_cfg.forecast.max_recursive_steps
    )
    assert "TRUNCATED" in curve.describe()


def test_recursion_responds_to_the_last_observed_level(
    sim_panel: pd.DataFrame, sim_cfg: Config, model: _StubModel
) -> None:
    """The feed-back has to be real.

    If the predictions were not written back into the panel, the recursive tail
    would be driven by climatology alone and would land in the same place whatever
    the recent history looked like.
    """
    lifted = sim_panel.copy()
    recent = lifted.index.get_level_values("date") >= pd.Timestamp("2017-06-01")
    lifted.loc[recent, "cases"] = lifted.loc[recent, "cases"] * 4

    base = _project(sim_panel, sim_cfg, model, months=4)
    shifted = _project(lifted, sim_cfg, model, months=4)

    assert (
        shifted.steps[-1].predicted_cases_per_100k
        != base.steps[-1].predicted_cases_per_100k
    )


def test_projection_does_not_mutate_the_panel(
    sim_panel: pd.DataFrame, sim_cfg: Config, model: _StubModel
) -> None:
    before = sim_panel.copy()
    _project(sim_panel, sim_cfg, model, months=4)
    pd.testing.assert_frame_equal(sim_panel, before)


def test_unknown_state_is_refused(
    sim_panel: pd.DataFrame, sim_cfg: Config, model: _StubModel
) -> None:
    with pytest.raises(SimulationError, match="Atlantis"):
        forecast_horizon(sim_panel, "Atlantis", date(2018, 1, 1), model, sim_cfg)


def test_state_with_no_observations_is_refused(
    sim_panel: pd.DataFrame, sim_cfg: Config, model: _StubModel
) -> None:
    """Degrading gracefully means saying why, not projecting from nothing."""
    blank = sim_panel.copy()
    blank.loc["Odisha", "cases"] = np.nan

    with pytest.raises(SimulationError, match="no observed"):
        forecast_horizon(blank, "Odisha", date(2018, 6, 1), model, sim_cfg)


def test_frame_and_at_expose_the_curve(
    sim_panel: pd.DataFrame, sim_cfg: Config, model: _StubModel
) -> None:
    curve = _project(sim_panel, sim_cfg, model, months=3)
    frame = curve.frame()

    assert list(frame.columns) == [
        "target_date", "steps_ahead", "predicted", "lower", "upper", "mode",
        "reliability",
    ]
    assert len(frame) == len(curve.steps)

    found = curve.at(curve.steps[-1].target_date)
    assert found is not None
    assert found.predicted_cases_per_100k == curve.steps[-1].predicted_cases_per_100k
    assert curve.at(pd.Timestamp("1999-01-01")) is None


def test_describe_carries_the_caveat(
    sim_panel: pd.DataFrame, sim_cfg: Config, model: _StubModel
) -> None:
    """The warning travels with the numbers or it does not exist."""
    text = _project(sim_panel, sim_cfg, model, months=4).describe()

    assert "RECURSIVE" in text
    assert "not a scenario" in text


def test_a_projection_runs_the_model_once_per_step(
    sim_panel: pd.DataFrame, sim_cfg: Config, model: _StubModel
) -> None:
    """Each step rebuilds the whole feature pipeline, so each one has to count.

    The value written back for a period is the one the previous iteration already
    predicted for it. Re-predicting to read a number already in hand rebuilt every
    lag, rolling window and spatial term for nothing, and nearly doubled the cost
    of a projection.
    """
    model.calls = 0
    curve = _project(sim_panel, sim_cfg, model, months=5)

    # One pass to reach the direct steps, then one per recursive step.
    recursive = sum(1 for step in curve.steps if step.is_recursive)
    assert model.calls == 1 + recursive


def test_climatological_normals_are_per_state_and_per_month(
    sim_panel: pd.DataFrame, sim_cfg: Config
) -> None:
    """A typical October means something different in each state."""
    normals = climatological_normals(sim_panel, sim_cfg)

    assert normals.index.names == ["state", "_period"]
    assert set(normals.index.get_level_values("_period")) == set(range(1, 13))

    kerala = sim_panel.loc["Kerala"]
    expected = kerala[kerala.index.month == 10]["rainfall"].mean()
    assert normals.loc[("Kerala", 10), "rainfall"] == pytest.approx(expected)
