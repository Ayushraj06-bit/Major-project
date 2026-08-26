"""Decision support: data-derived thresholds, upper-bound alerting, traceability."""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import ActionSet, Config, ConfigError
from src.panel import complete_index
from src.recommend import (
    BASELINE_LABEL,
    RecommendationError,
    Threshold,
    alert_summary,
    assign_tier,
    compute_thresholds,
    recommend,
    render,
    render_all,
    to_frame,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def rec_panel(cfg: Config) -> pd.DataFrame:
    """A panel with a clear seasonal outbreak, so thresholds have something to find."""
    index = complete_index(cfg)
    month = index.get_level_values("date").month.to_numpy()
    rng = np.random.default_rng(31)
    seasonal = np.clip(np.sin(2 * np.pi * (month - 7) / 12), 0, None) ** 3
    return pd.DataFrame(
        {
            "cases": 5 + 400 * seasonal + rng.normal(0, 2, len(index)).clip(0),
            "population": 3.0e7,
        },
        index=index,
    )


def _forecasts(states: list[str], values: list[float], width: float = 0.2) -> pd.DataFrame:
    """A minimal forecast frame in the shape ProductionModel.predict returns."""
    frame = pd.DataFrame(
        {
            "state": states,
            "origin_date": [pd.Timestamp("2017-08-01")] * len(states),
            "predicted_log": values,
        }
    )
    frame["target_date"] = frame["origin_date"] + pd.DateOffset(months=1)
    frame["lower_log"] = frame["predicted_log"] - width
    frame["upper_log"] = frame["predicted_log"] + width
    for column in ("predicted", "lower", "upper"):
        frame[f"{column}_cases_per_100k"] = np.expm1(frame[f"{column}_log"])
    return frame


# --------------------------------------------------------------------------- #
# Review gate: thresholds are derived, not hand-picked
# --------------------------------------------------------------------------- #


def test_no_hardcoded_threshold_numbers_in_the_module() -> None:
    """The gate: an if-else table is exactly what this component must not be.

    Parsed rather than grepped. Any bare numeric literal compared against a value
    would be a hand-picked threshold; the only constants allowed are structural.
    """
    tree = ast.parse((PROJECT_ROOT / "src" / "recommend.py").read_text(encoding="utf-8"))
    allowed = {0, 1, 2}

    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            for operand in [node.left, *node.comparators]:
                if isinstance(operand, ast.Constant) and isinstance(
                    operand.value, int | float
                ):
                    assert operand.value in allowed, (
                        f"recommend.py compares against the literal {operand.value}; "
                        "thresholds must come from config or from the data"
                    )


def test_thresholds_come_from_each_states_own_history(
    rec_panel: pd.DataFrame, cfg: Config
) -> None:
    """Per state, because the same rate means different things in different places."""
    louder = rec_panel.copy()
    # Unambiguous MultiIndex form: louder.loc["Kerala", "cases"] reads as
    # (row, column) and the assignment back does not align.
    louder.loc[("Kerala", slice(None)), "cases"] *= 4.0

    thresholds = compute_thresholds(louder, cfg)
    kerala = thresholds.for_state("Kerala")[-1]
    odisha = thresholds.for_state("Odisha")[-1]
    assert kerala.value_log > odisha.value_log


def test_each_threshold_records_the_evidence_behind_it(
    rec_panel: pd.DataFrame, cfg: Config
) -> None:
    """'Why that number?' must have an answer attached to the number."""
    thresholds = compute_thresholds(rec_panel, cfg)
    for threshold in thresholds.for_state("Kerala"):
        assert "percentile" in threshold.label
        assert threshold.n_observations > 0
        assert np.isfinite(threshold.value_log)
        assert threshold.value_cases_per_100k == pytest.approx(
            np.expm1(threshold.value_log)
        )


def test_quantile_thresholds_are_ordered_by_tier(
    rec_panel: pd.DataFrame, cfg: Config
) -> None:
    thresholds = compute_thresholds(rec_panel, cfg)
    values = [threshold.value_log for threshold in thresholds.for_state("Kerala")]
    assert values == sorted(values)


def test_ewma_method_is_available_as_an_alternative(
    rec_panel: pd.DataFrame, cfg: Config
) -> None:
    """The established outbreak-detection form, selectable from config."""
    ewma_cfg = dataclasses.replace(cfg, risk=dataclasses.replace(cfg.risk, method="ewma"))
    thresholds = compute_thresholds(rec_panel, ewma_cfg)

    assert thresholds.method == "ewma"
    for threshold in thresholds.for_state("Kerala"):
        assert "EWMA" in threshold.label


def test_unknown_method_is_refused_and_names_farrington(
    rec_panel: pd.DataFrame, cfg: Config
) -> None:
    """Better to refuse than ship a half-implemented Farrington."""
    with pytest.raises(ConfigError, match="risk.method"):
        dataclasses.replace(cfg.risk, method="farrington")


def test_a_state_without_history_is_refused_not_defaulted(
    rec_panel: pd.DataFrame, cfg: Config
) -> None:
    thresholds = compute_thresholds(rec_panel, cfg)
    with pytest.raises(RecommendationError, match="no thresholds for"):
        thresholds.for_state("Atlantis")


# --------------------------------------------------------------------------- #
# Review gate: alerting on the upper bound
# --------------------------------------------------------------------------- #


def test_tier_is_assigned_on_the_upper_bound_not_the_point_forecast(
    rec_panel: pd.DataFrame, cfg: Config
) -> None:
    """The gate: a point forecast below the threshold must still alert.

    Its interval reaches above the boundary, and preparedness turns on the
    plausible worst case rather than the expectation.
    """
    thresholds = compute_thresholds(rec_panel, cfg)
    boundary = thresholds.for_state("Kerala")[-1].value_log

    just_below = _forecasts(["Kerala"], [boundary - 0.05], width=0.2)
    [recommendation] = recommend(just_below, thresholds, cfg)

    assert recommendation.predicted_log < boundary
    assert recommendation.trigger_value_log > boundary
    assert recommendation.tier == cfg.risk.tiers[-1]
    assert "upper bound" in recommendation.trigger_basis


def test_point_forecast_basis_is_selectable_and_alerts_less(
    rec_panel: pd.DataFrame, cfg: Config
) -> None:
    thresholds = compute_thresholds(rec_panel, cfg)
    boundary = thresholds.for_state("Kerala")[-1].value_log
    frame = _forecasts(["Kerala"], [boundary - 0.05], width=0.2)

    point_cfg = dataclasses.replace(cfg, risk=dataclasses.replace(cfg.risk, alert_on="point"))
    [on_point] = recommend(frame, thresholds, point_cfg)
    assert on_point.tier != cfg.risk.tiers[-1]
    assert on_point.trigger_basis == "point forecast"


def test_missing_interval_is_refused_when_alerting_on_the_upper_bound(
    rec_panel: pd.DataFrame, cfg: Config
) -> None:
    """Silently falling back to the point forecast would alert late without saying so."""
    thresholds = compute_thresholds(rec_panel, cfg)
    frame = _forecasts(["Kerala"], [0.5]).drop(columns=["upper_log"])
    with pytest.raises(RecommendationError, match="ConformalForecaster"):
        recommend(frame, thresholds, cfg)


# --------------------------------------------------------------------------- #
# Review gate: every recommendation names its trigger
# --------------------------------------------------------------------------- #


def test_every_recommendation_names_the_number_that_triggered_it(
    rec_panel: pd.DataFrame, cfg: Config
) -> None:
    """The gate, as one assertion per alert."""
    thresholds = compute_thresholds(rec_panel, cfg)
    high = thresholds.for_state("Kerala")[-1].value_log
    frame = _forecasts(["Kerala", "Odisha"], [high + 0.5, -5.0])

    for recommendation in recommend(frame, thresholds, cfg):
        evidence = recommendation.evidence()
        assert np.isfinite(evidence["trigger_value_cases_per_100k"])
        assert evidence["trigger_basis"]
        if recommendation.is_alert:
            assert evidence["threshold_value_cases_per_100k"] is not None
            assert evidence["threshold_label"]
            assert evidence["threshold_observations"] > 0
        else:
            assert evidence["threshold_label"] == BASELINE_LABEL


def test_the_highest_crossed_boundary_is_the_one_quoted(cfg: Config) -> None:
    """Quoting the first boundary passed would understate the justification."""
    thresholds = (
        Threshold("Kerala", "MEDIUM", 1.0, np.expm1(1.0), "75% historical percentile", 100),
        Threshold("Kerala", "HIGH", 2.0, np.expm1(2.0), "90% historical percentile", 100),
    )
    tier, crossed = assign_tier(3.0, thresholds, cfg)
    assert tier == "HIGH"
    assert crossed is not None and crossed.label.startswith("90%")


def test_below_every_boundary_lands_in_the_baseline_tier(cfg: Config) -> None:
    thresholds = (
        Threshold("Kerala", "HIGH", 2.0, np.expm1(2.0), "90% historical percentile", 100),
    )
    tier, crossed = assign_tier(-1.0, thresholds, cfg)
    assert tier == cfg.risk.tiers[0]
    assert crossed is None


# --------------------------------------------------------------------------- #
# Review gate: the action mapping lives in config
# --------------------------------------------------------------------------- #


def test_actions_come_from_config_not_code() -> None:
    """The gate: a domain expert must be able to revise these without a developer."""
    tree = ast.parse((PROJECT_ROOT / "src" / "recommend.py").read_text(encoding="utf-8"))

    # Docstrings quote the target output verbatim, which is documentation rather
    # than a hardcoded action, so they are excluded.
    docstrings = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
    }

    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docstrings
        ):
            lowered = node.value.lower()
            assert "source-reduction" not in lowered, "action text belongs in config.yaml"
            assert "bed pre-positioning" not in lowered


def test_recommendation_carries_the_configured_actions(
    rec_panel: pd.DataFrame, cfg: Config
) -> None:
    thresholds = compute_thresholds(rec_panel, cfg)
    frame = _forecasts(["Kerala"], [thresholds.for_state("Kerala")[-1].value_log + 1.0])
    [recommendation] = recommend(frame, thresholds, cfg)

    assert recommendation.actions == cfg.risk.actions_for(recommendation.tier)
    assert recommendation.action_source == cfg.risk.action_source


def test_a_tier_with_no_actions_is_refused_at_startup(cfg: Config) -> None:
    """A recommendation recommending nothing is worse than a config error."""
    with pytest.raises(ConfigError, match="no actions defined for tier"):
        dataclasses.replace(
            cfg.risk, actions=(ActionSet(tier="LOW", actions=("do something",)),)
        )


def test_an_action_set_for_an_unknown_tier_is_refused(cfg: Config) -> None:
    with pytest.raises(ConfigError, match="not in risk.tiers"):
        dataclasses.replace(
            cfg.risk,
            actions=(*cfg.risk.actions, ActionSet(tier="CRITICAL", actions=("evacuate",))),
        )


# --------------------------------------------------------------------------- #
# Structure and rendering
# --------------------------------------------------------------------------- #


def test_recommendations_stay_machine_readable(
    rec_panel: pd.DataFrame, cfg: Config
) -> None:
    """The dashboard sorts and colours on fields, not on a parsed sentence."""
    thresholds = compute_thresholds(rec_panel, cfg)
    frame = _forecasts(["Kerala", "Odisha"], [2.0, -5.0])
    table = to_frame(recommend(frame, thresholds, cfg))

    assert {"state", "tier", "trigger_value_cases_per_100k", "threshold_label"} <= set(
        table.columns
    )
    assert len(table) == 2


def test_render_produces_the_target_sentence(rec_panel: pd.DataFrame, cfg: Config) -> None:
    thresholds = compute_thresholds(rec_panel, cfg)
    high = thresholds.for_state("Kerala")[-1]
    frame = _forecasts(["Kerala"], [high.value_log + 1.0])

    drivers = {
        ("Kerala", pd.Timestamp("2017-08-01")): (
            ("rainfall lag-3", 0.4), ("humidity lag-2", 0.2)
        )
    }
    [recommendation] = recommend(frame, thresholds, cfg, drivers=drivers)
    text = render(recommendation)

    assert text.startswith("HIGH - predicted ")
    assert "interval upper bound" in text
    assert "90% historical percentile" in text
    assert "Top drivers: rainfall lag-3, humidity lag-2." in text
    assert "Actions: " in text


def test_render_omits_drivers_rather_than_inventing_them(
    rec_panel: pd.DataFrame, cfg: Config
) -> None:
    """Attributions are computed for a sample of rows, not all of them."""
    thresholds = compute_thresholds(rec_panel, cfg)
    frame = _forecasts(["Kerala"], [thresholds.for_state("Kerala")[-1].value_log + 1.0])
    [recommendation] = recommend(frame, thresholds, cfg)

    assert recommendation.drivers == ()
    assert "Top drivers" not in render(recommendation)


def test_render_all_puts_the_worst_first(rec_panel: pd.DataFrame, cfg: Config) -> None:
    thresholds = compute_thresholds(rec_panel, cfg)
    high = thresholds.for_state("Kerala")[-1].value_log
    frame = _forecasts(["Kerala", "Odisha"], [high + 2.0, -5.0])

    rendered = render_all(recommend(frame, thresholds, cfg))
    assert rendered.splitlines()[0].startswith("HIGH")


# --------------------------------------------------------------------------- #
# The alert-rate diagnostic
# --------------------------------------------------------------------------- #


def test_alert_summary_exposes_the_upper_bound_over_alerting(
    rec_panel: pd.DataFrame, cfg: Config
) -> None:
    """Realised rates exceed nominal because trigger and threshold differ in kind.

    Thresholds are quantiles of the observed distribution; tiers are assigned on
    the interval upper bound. That is deliberate, and the gap is worth surfacing
    rather than leaving a reader to assume the top tier fires one time in ten.
    """
    thresholds = compute_thresholds(rec_panel, cfg)
    boundary = thresholds.for_state("Kerala")[-1].value_log
    frame = _forecasts(["Kerala"] * 10, [boundary - 0.05] * 10, width=0.2)

    summary = alert_summary(recommend(frame, thresholds, cfg), cfg)
    top = summary[summary["tier"] == cfg.risk.tiers[-1]].iloc[0]

    assert top["realised_share"] > top["nominal_share"]
    assert set(summary["tier"]) == set(cfg.risk.tiers)


def test_alert_summary_needs_something_to_summarise(cfg: Config) -> None:
    with pytest.raises(RecommendationError, match="no recommendations"):
        alert_summary([], cfg)
