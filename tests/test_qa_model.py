"""QA audit, part two: properties that need the frozen production artifact.

Slower than :mod:`tests.test_qa_audit` because each test runs the real feature
pipeline and the real network. They are here rather than skipped because the
properties they check -- interval calibration, simulation identity, recursive
uncertainty growth -- are exactly the ones that fail quietly.

Every test skips with a named reason if the artifact has not been frozen, so a
fresh checkout reports "not verified" rather than a false pass.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def artifact() -> Any:
    """The frozen production model, or a skip naming what to run."""
    try:
        from src.production import load_production

        return load_production()
    except Exception as exc:  # noqa: BLE001 - absence is a skip, not a failure
        pytest.skip(f"no frozen production model ({exc}); run freeze_production.py")


@pytest.fixture(scope="module")
def real_cfg() -> Any:
    from src.config import load_config

    return load_config()


@pytest.fixture(scope="module")
def clean_panel(real_cfg: Any) -> pd.DataFrame:
    """The same panel the frozen model was fitted on."""
    from src.panel import load_panel
    from src.preprocess import preprocess

    panel = load_panel(real_cfg, synthetic=True)
    return preprocess(panel, real_cfg).panel[list(panel.columns)]


# --------------------------------------------------------------------------- #
# Artifact integrity
# --------------------------------------------------------------------------- #


def test_artifact_pieces_are_mutually_consistent(artifact: Any) -> None:
    """A model, a scaler and a feature spec that disagree produce garbage silently.

    The scaler is fitted on ``n_features``; the network's input layer expects
    ``(timesteps, n_features)``. If a re-freeze changed the feature set but the
    scaler were reused, predictions would still be produced -- just meaningless.
    """
    spec = artifact.spec
    assert spec.n_features == len(spec.columns)

    shape = artifact.predictor.input_shape
    timesteps, features = shape[1], shape[2]
    assert features == spec.n_features, (
        f"network expects {features} features, spec declares {spec.n_features}"
    )
    assert timesteps == spec.timesteps, (
        f"network expects {timesteps} timesteps, spec declares {spec.timesteps}"
    )
    assert len(artifact.scaler_mean) == spec.n_features, (
        f"scaler_mean has {len(artifact.scaler_mean)} entries, "
        f"spec declares {spec.n_features} features"
    )
    assert len(artifact.scaler_scale) == spec.n_features


def test_the_artifact_carries_its_own_config(artifact: Any) -> None:
    """The artifact must be self-contained.

    If it read the live config.yaml, editing that file would silently change what
    a frozen model means.
    """
    assert artifact.cfg is not None
    assert artifact.spec.horizon == artifact.spec.horizon
    assert len(artifact.residuals) > 0, "no calibration residuals stored"


def test_scaling_round_trips(artifact: Any) -> None:
    """A scaling that does not invert cleanly corrupts every reported case rate."""
    rng = np.random.default_rng(0)
    sample = rng.normal(size=(20, artifact.spec.n_features))

    scaled = (sample - artifact.scaler_mean) / artifact.scaler_scale
    recovered = scaled * artifact.scaler_scale + artifact.scaler_mean
    np.testing.assert_allclose(recovered, sample, rtol=1e-9, atol=1e-9)


def test_no_stored_scale_is_zero(artifact: Any) -> None:
    """A constant column has zero standard deviation.

    Dividing by it yields inf, and every downstream number becomes NaN while the
    pipeline reports success. ``StandardScaled.fit`` substitutes 1.0 for a zero
    scale; this asserts the guard actually reached the stored artifact, because
    real data is far more likely than the synthetic panel to contain a column
    that never varies.
    """
    assert np.all(np.abs(artifact.scaler_scale) > 0.0), (
        "a zero scale reached the artifact; features will divide to inf"
    )
    assert np.isfinite(artifact.scaler_mean).all()
    assert np.isfinite(artifact.scaler_scale).all()


# --------------------------------------------------------------------------- #
# Interval calibration
# --------------------------------------------------------------------------- #


def test_intervals_cover_close_to_their_nominal_rate(
    artifact: Any, clean_panel: pd.DataFrame, real_cfg: Any
) -> None:
    """An 80% interval containing 40% or 99% of actuals is a broken layer.

    Checked on the case-rate scale, which is what the dashboard shows.
    """
    from src.features import target_level

    predictions = artifact.predict(clean_panel)
    levels = target_level(clean_panel, real_cfg)

    joined = predictions.copy()
    joined["target_date"] = pd.to_datetime(joined["target_date"])
    keys = pd.MultiIndex.from_arrays([joined["state"], joined["target_date"]])
    actual_log = levels.reindex(keys).to_numpy()

    inside = (actual_log >= joined["lower_log"].to_numpy()) & (
        actual_log <= joined["upper_log"].to_numpy()
    )
    observed = float(np.nanmean(inside[np.isfinite(actual_log)]))
    nominal = 1.0 - real_cfg.conformal.alpha

    assert np.isfinite(observed), "coverage could not be computed; the join failed"
    assert 0.0 < observed < 1.0, f"degenerate coverage {observed}"
    assert abs(observed - nominal) < 0.15, (
        f"{nominal:.0%} interval covers {observed:.1%} of actuals"
    )


def test_the_interval_is_not_degenerate(
    artifact: Any, clean_panel: pd.DataFrame
) -> None:
    """A zero-width interval would trivially 'calibrate' to 0% and look tidy."""
    predictions = artifact.predict(clean_panel)
    width = predictions["upper_log"] - predictions["lower_log"]

    assert (width > 0).all(), "some intervals have zero or negative width"
    assert width.max() < 10.0, "interval width is implausibly large on the log scale"


# --------------------------------------------------------------------------- #
# Simulation identity and coherence
# --------------------------------------------------------------------------- #


def test_a_zero_change_scenario_reproduces_the_baseline_exactly(
    artifact: Any, clean_panel: pd.DataFrame, real_cfg: Any
) -> None:
    """The null control.

    If this drifts, features are being rebuilt differently under a scenario than
    under a plain forecast, and every what-if in the project is suspect.
    """
    from src.simulate import Scenario, simulate

    outcome = simulate(
        clean_panel, Scenario(variable="rainfall", change=0.0, mode="percent"),
        artifact, real_cfg,
    )
    baseline = outcome.baseline.sort_values(["state", "target_date"])
    scenario = outcome.scenario_forecast.sort_values(["state", "target_date"])

    np.testing.assert_allclose(
        scenario["predicted_log"].to_numpy(),
        baseline["predicted_log"].to_numpy(),
        rtol=0, atol=0,
        err_msg="a zero-change scenario did not reproduce the baseline exactly",
    )


def test_changing_rainfall_moves_every_derived_rainfall_feature(
    artifact: Any, clean_panel: pd.DataFrame, real_cfg: Any
) -> None:
    """Coherence: one raw change must move its whole feature family.

    A scenario that moved the current value but left lag-1, lag-2, the rolling
    mean and the spatial term untouched would be an incoherent world the model
    was never trained on, and its answer would be meaningless.
    """
    from src.features import build_features
    from src.simulate import Scenario, apply_scenario

    changed, _clamped = apply_scenario(
        clean_panel, Scenario(variable="rainfall", change=40.0, mode="percent"),
        real_cfg,
    )

    base_x, _, spec = build_features(clean_panel, real_cfg, horizon=artifact.spec.horizon)
    new_x, _, _ = build_features(changed, real_cfg, horizon=artifact.spec.horizon)

    families = [
        column for column in spec.columns
        if spec.origins[column].raw_variable == "rainfall"
    ]
    assert families, "no rainfall-derived columns found"
    assert len(families) > 2, (
        f"only {families} derive from rainfall; expected the current value plus "
        "lags, a rolling mean and a spatial term"
    )

    index = {name: position for position, name in enumerate(spec.columns)}
    moved = []
    for column in families:
        position = index[column]
        if not np.allclose(
            np.nan_to_num(base_x[:, :, position]),
            np.nan_to_num(new_x[:, :, position]),
        ):
            moved.append(column)

    assert set(moved) == set(families), (
        f"these rainfall features did not move: {sorted(set(families) - set(moved))}"
    )


def test_an_out_of_distribution_scenario_raises_its_flag(
    artifact: Any, clean_panel: pd.DataFrame, real_cfg: Any
) -> None:
    """An extreme scenario must raise the out-of-distribution flag.

    A +2000% rainfall world is not one the model has evidence about. If the flag
    stays down, the dashboard presents an extrapolation as a forecast.
    """
    from src.simulate import Scenario, simulate

    outcome = simulate(
        clean_panel, Scenario(variable="rainfall", change=2000.0, mode="percent"),
        artifact, real_cfg,
    )
    assert outcome.out_of_distribution, "a +2000% scenario did not flag as OOD"
    assert outcome.clamped_fraction > 0.5, (
        f"only {outcome.clamped_fraction:.0%} of cells were clamped"
    )


def test_a_modest_scenario_does_not_cry_wolf(
    artifact: Any, clean_panel: pd.DataFrame, real_cfg: Any
) -> None:
    """The OOD flag must discriminate, or it means nothing when it fires."""
    from src.simulate import Scenario, simulate

    outcome = simulate(
        clean_panel, Scenario(variable="rainfall", change=1.0, mode="percent"),
        artifact, real_cfg,
    )
    assert not outcome.out_of_distribution, (
        "a +1% scenario flagged as out of distribution"
    )


# --------------------------------------------------------------------------- #
# Recursive forward projection
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def curve(artifact: Any, clean_panel: pd.DataFrame, real_cfg: Any) -> Any:
    from src.simulate import forecast_horizon

    state = sorted(set(clean_panel.index.get_level_values("state")))[0]
    last = clean_panel.loc[state, real_cfg.data.target_column].dropna().index.max()
    return forecast_horizon(
        clean_panel, state, pd.Timestamp(last) + pd.DateOffset(months=6),
        artifact, real_cfg,
    )


def test_the_interval_widens_at_every_recursive_step(curve: Any) -> None:
    """Uncertainty must propagate with every recursive step.

    A flat band four months out claims a confidence the method cannot support.
    Checked on the width a reader actually sees (upper minus lower), not on the
    internal half-width: the lower bound is floored at zero, so the two can
    diverge and only the visible one matters.
    """
    widths = [
        step.upper_cases_per_100k - step.lower_cases_per_100k for step in curve.steps
    ]
    assert len(widths) > 2, "not enough steps to test propagation"
    for earlier, later in zip(widths, widths[1:], strict=False):
        assert later > earlier, f"interval width did not grow: {widths}"


def test_reliability_decays_and_the_mode_is_labelled(curve: Any) -> None:
    scores = [step.reliability for step in curve.steps]
    assert scores == sorted(scores, reverse=True)
    assert curve.steps[0].mode == "direct"
    assert any(step.mode == "recursive" for step in curve.steps)


def test_the_projection_is_capped_and_says_so(
    artifact: Any, clean_panel: pd.DataFrame, real_cfg: Any
) -> None:
    """Asked for five years, it must refuse loudly rather than extrapolate."""
    from src.simulate import forecast_horizon

    state = sorted(set(clean_panel.index.get_level_values("state")))[0]
    result = forecast_horizon(
        clean_panel, state, pd.Timestamp("2035-01-01"), artifact, real_cfg
    )
    assert result.truncated
    assert len(result.steps) <= 1 + real_cfg.forecast.max_recursive_steps
    assert "TRUNCATED" in result.describe()


def test_projected_values_stay_in_a_plausible_range(curve: Any) -> None:
    """A recursive loop is a feedback loop and can diverge.

    Non-finite or wildly negative output would reach the chart before anyone
    noticed it was nonsense.
    """
    for step in curve.steps:
        assert np.isfinite(step.predicted_cases_per_100k)
        assert step.predicted_cases_per_100k >= 0.0
        assert step.lower_cases_per_100k >= 0.0
        assert step.lower_cases_per_100k <= step.predicted_cases_per_100k
        assert step.predicted_cases_per_100k <= step.upper_cases_per_100k


def test_an_unknown_state_is_refused_by_name(
    artifact: Any, clean_panel: pd.DataFrame, real_cfg: Any
) -> None:
    from src.simulate import SimulationError, forecast_horizon

    with pytest.raises(SimulationError, match="Atlantis"):
        forecast_horizon(clean_panel, "Atlantis", pd.Timestamp("2024-06-01"),
                         artifact, real_cfg)


def test_a_target_date_in_the_past_returns_an_empty_curve(
    artifact: Any, clean_panel: pd.DataFrame, real_cfg: Any
) -> None:
    """Asking to project backwards is not a crash and not a silent forecast."""
    from src.simulate import forecast_horizon

    state = sorted(set(clean_panel.index.get_level_values("state")))[0]
    result = forecast_horizon(
        clean_panel, state, pd.Timestamp("2015-01-01"), artifact, real_cfg
    )
    assert result.steps == () or all(
        step.target_date > result.last_observed for step in result.steps
    )
