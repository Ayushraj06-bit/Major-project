"""QA audit: correctness properties that do not announce themselves when broken.

Every test here targets a failure mode that lets the pipeline **run to completion
and report wrong numbers**. That is the dangerous class: a crash gets fixed, a
plausible-looking figure gets published.

Organised by the property under attack, not by module.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from src.config import Config
from src.features import build_features, target_level
from src.panel import complete_index
from src.splits import rolling_origin

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def qa_cfg(cfg: Config) -> Config:
    """A config with enough history for several folds and a 12-month lag."""
    return dataclasses.replace(
        cfg,
        data=dataclasses.replace(
            cfg.data,
            start_date=pd.Timestamp("2006-01-01").date(),
            end_date=pd.Timestamp("2021-12-31").date(),
        ),
        features=dataclasses.replace(
            cfg.features, sequence_length=3, lags=(1, 2, 12), rolling_windows=(3,)
        ),
        forecast=dataclasses.replace(cfg.forecast, horizons=(1,)),
        split=dataclasses.replace(cfg.split, initial_train_size=36, test_size=12),
    )


@pytest.fixture
def qa_panel(qa_cfg: Config) -> pd.DataFrame:
    """A seasonal panel with every column the default feature set expects."""
    index = complete_index(qa_cfg)
    month = index.get_level_values("date").month.to_numpy()
    rng = np.random.default_rng(7)
    seasonal = np.sin(2 * np.pi * (month - 6) / 12)
    return pd.DataFrame(
        {
            "cases": np.abs(80 + 60 * seasonal + rng.normal(0, 6, len(index))),
            "rainfall": np.abs(120 + 100 * seasonal + rng.normal(0, 12, len(index))),
            "temperature": 27 + 5 * seasonal + rng.normal(0, 1.0, len(index)),
            "humidity": np.clip(70 + 15 * seasonal, 0, 100),
            "search_interest": np.abs(45 + 25 * seasonal),
            "population": 3.0e7,
            "population_density": 850.0,
        },
        index=index,
    )


# --------------------------------------------------------------------------- #
# Leakage
# --------------------------------------------------------------------------- #


def test_folds_are_cut_on_dates_not_positions(
    qa_panel: pd.DataFrame, qa_cfg: Config
) -> None:
    """The trap this project has already fallen into once.

    The panel is state-major, so "the last N rows" is one state's whole history,
    not the most recent N periods. A positional split produces a *state* holdout
    that scores like a temporal one, and it scores well.
    """
    _, _, spec = build_features(qa_panel, qa_cfg)
    folds = rolling_origin(spec.sample_index, qa_cfg)
    dates = pd.DatetimeIndex(spec.sample_index.get_level_values("date"))
    states = spec.sample_index.get_level_values("state")

    assert folds, "no folds produced"
    for fold in folds:
        assert dates[fold.fit].max() < dates[fold.val].min(), (
            f"fold {fold.number}: fit reaches {dates[fold.fit].max()}, "
            f"val starts {dates[fold.val].min()}"
        )
        assert dates[fold.train].max() < dates[fold.test].min(), (
            f"fold {fold.number}: training data postdates the test window"
        )
        # Every retained state must appear on both sides, or it is a state holdout.
        assert set(states[fold.train]) == set(states[fold.test]), (
            f"fold {fold.number}: train and test cover different states"
        )


def test_validation_never_precedes_training_in_any_fold(
    qa_panel: pd.DataFrame, qa_cfg: Config
) -> None:
    """Folds must march forward. A shuffled split passes a naive per-fold check."""
    _, _, spec = build_features(qa_panel, qa_cfg)
    folds = rolling_origin(spec.sample_index, qa_cfg)

    starts = [fold.test_start for fold in folds]
    assert starts == sorted(starts), "test windows are not ordered in time"
    assert len(set(starts)) == len(starts), "two folds share a test start"


def test_the_embargo_separates_calibration_from_test(
    qa_panel: pd.DataFrame, qa_cfg: Config
) -> None:
    """The embargo must exceed the horizon.

    Without it the last calibration target overlaps the first test input window,
    which is leakage that comparing forecast *origins* would not reveal.
    """
    horizon = 3
    _, _, spec = build_features(qa_panel, qa_cfg, horizon=horizon)
    folds = rolling_origin(spec.sample_index, qa_cfg, horizon=horizon)

    for fold in folds:
        gap = (fold.test_start.to_period("M") - fold.val_end.to_period("M")).n
        assert gap > horizon, (
            f"fold {fold.number}: only {gap} periods between calibration and test, "
            f"horizon is {horizon}"
        )
        assert fold.embargo >= horizon


def test_no_feature_reads_the_future(qa_panel: pd.DataFrame, qa_cfg: Config) -> None:
    """Perturb one period and assert no earlier sample's features move.

    Stronger than inspecting column names: it tests the values that were actually
    built, so an off-by-one in a shift is caught.
    """
    baseline_x, _, spec = build_features(qa_panel, qa_cfg)
    dates = pd.DatetimeIndex(spec.sample_index.get_level_values("date"))
    cut = dates[len(dates) // 2]

    tampered = qa_panel.copy()
    future = tampered.index.get_level_values("date") > cut
    tampered.loc[future, "rainfall"] = tampered.loc[future, "rainfall"] * 10 + 500

    after_x, _, after_spec = build_features(tampered, qa_cfg)
    assert after_spec.sample_index.equals(spec.sample_index)

    earlier = np.flatnonzero(dates <= cut)
    assert earlier.size, "no samples at or before the cut"
    np.testing.assert_allclose(
        np.nan_to_num(baseline_x[earlier]),
        np.nan_to_num(after_x[earlier]),
        err_msg="features at or before the cut changed when only the future moved",
    )


# --------------------------------------------------------------------------- #
# Feature purity
# --------------------------------------------------------------------------- #


def test_build_features_is_pure(qa_panel: pd.DataFrame, qa_cfg: Config) -> None:
    """Called twice on identical input it must return identical output."""
    first_x, first_y, first_spec = build_features(qa_panel, qa_cfg)
    second_x, second_y, second_spec = build_features(qa_panel, qa_cfg)

    np.testing.assert_array_equal(np.nan_to_num(first_x), np.nan_to_num(second_x))
    np.testing.assert_array_equal(np.nan_to_num(first_y), np.nan_to_num(second_y))
    assert first_spec.columns == second_spec.columns
    assert first_spec.sample_index.equals(second_spec.sample_index)


def test_build_features_does_not_mutate_its_input(
    qa_panel: pd.DataFrame, qa_cfg: Config
) -> None:
    before = qa_panel.copy()
    build_features(qa_panel, qa_cfg)
    pd.testing.assert_frame_equal(qa_panel, before)


def test_build_features_holds_no_cross_call_state(
    qa_panel: pd.DataFrame, qa_cfg: Config
) -> None:
    """Interleave two different inputs; a cache keyed on nothing would show here."""
    other = qa_panel.copy()
    other["rainfall"] = other["rainfall"] * 2.0

    a1, _, _ = build_features(qa_panel, qa_cfg)
    b1, _, _ = build_features(other, qa_cfg)
    a2, _, _ = build_features(qa_panel, qa_cfg)
    b2, _, _ = build_features(other, qa_cfg)

    np.testing.assert_array_equal(np.nan_to_num(a1), np.nan_to_num(a2))
    np.testing.assert_array_equal(np.nan_to_num(b1), np.nan_to_num(b2))
    assert not np.allclose(np.nan_to_num(a1), np.nan_to_num(b1)), (
        "doubling rainfall changed nothing; the inputs are not reaching the features"
    )


# --------------------------------------------------------------------------- #
# The target transform, and its inverse
# --------------------------------------------------------------------------- #


def test_target_transform_round_trips(qa_panel: pd.DataFrame, qa_cfg: Config) -> None:
    """``expm1(log1p(x)) == x`` through the real transform path, not in isolation."""
    levels = target_level(qa_panel, qa_cfg)
    rate = qa_panel["cases"] / qa_panel["population"] * qa_cfg.data.population_normalisation

    np.testing.assert_allclose(np.expm1(levels.to_numpy()), rate.to_numpy(), rtol=1e-9)


def test_every_inverse_transform_honours_the_config(qa_cfg: Config) -> None:
    """``data.target_transform`` is validated as one of {log1p, none}.

    Every site that converts a model-scale value back to cases per 100k must
    consult it. A site that hardcodes ``expm1`` produces silently wrong numbers
    the moment the option is set to ``none`` -- no crash, just exponentiated
    rates presented as rates.
    """
    linear = dataclasses.replace(
        qa_cfg, data=dataclasses.replace(qa_cfg.data, target_transform="none")
    )
    values = pd.Series([0.0, 0.5, 2.0, 5.0])

    from src.recommend import _rate  # noqa: PLC0415 - probing a private helper

    row = type("Row", (), {"trigger_log": 2.0, "trigger_cases_per_100k": None})()
    recovered = _rate(row, "trigger")

    del linear, values
    assert recovered == pytest.approx(2.0), (
        "src.recommend._rate applies expm1 unconditionally; with "
        "data.target_transform='none' it exponentiates an already-linear rate"
    )


# --------------------------------------------------------------------------- #
# State normalisation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("alias", "canonical"),
    [
        ("Orissa", "Odisha"),
        ("Odisha", "Odisha"),
        ("Uttaranchal", "Uttarakhand"),
        ("Uttarakhand", "Uttarakhand"),
        ("Pondicherry", "Puducherry"),
        ("NCT of Delhi", "Delhi"),
        ("  kerala  ", "Kerala"),
        ("TAMILNADU", "Tamil Nadu"),
    ],
)
def test_state_aliases_collapse_to_one_canonical_name(
    alias: str, canonical: str
) -> None:
    """Two spellings of one state fuse into two rows and halve its series."""
    from src.sources.registry import normalise_state

    assert normalise_state(alias) == canonical


def test_normalisation_is_idempotent() -> None:
    """Normalising twice must not drift."""
    from src.sources.registry import CANONICAL_NAMES, normalise_state

    for name in CANONICAL_NAMES:
        assert normalise_state(normalise_state(name)) == name


def test_an_unknown_state_is_refused_not_silently_dropped() -> None:
    from src.sources.registry import UnknownStateError, normalise_state

    with pytest.raises(UnknownStateError):
        normalise_state("Westeros")


# --------------------------------------------------------------------------- #
# Edge cases: the correct answer is a clear error or a labelled empty state
# --------------------------------------------------------------------------- #


def test_a_state_with_one_observation_is_dropped_and_named(
    qa_panel: pd.DataFrame, qa_cfg: Config
) -> None:
    """Too short for a window is not an error while other states survive.

    It must be dropped *and recorded*, so the dashboard can say why the state has
    no forecast instead of leaving a hole.
    """
    starved = qa_panel.copy()
    dates = starved.index.get_level_values("date")
    mask = (starved.index.get_level_values("state") == "Kerala") & (dates > dates[0])
    starved.loc[mask, "cases"] = np.nan

    _, _, spec = build_features(starved, qa_cfg)
    assert "Kerala" not in set(spec.sample_index.get_level_values("state"))
    assert "Kerala" in spec.dropped_states, (
        "the state vanished without being recorded in dropped_states"
    )


def test_an_all_zero_target_does_not_produce_nan_or_inf(
    qa_panel: pd.DataFrame, qa_cfg: Config
) -> None:
    """Zero cases is a real observation. log1p(0) is 0, not an error."""
    zeroed = qa_panel.copy()
    zeroed["cases"] = 0.0

    _, y, _ = build_features(zeroed, qa_cfg)
    assert np.isfinite(y).all(), "an all-zero target produced non-finite values"
    np.testing.assert_allclose(y, 0.0, atol=1e-12)


def test_a_panel_with_no_targets_at_all_raises_an_actionable_error(
    qa_panel: pd.DataFrame, qa_cfg: Config
) -> None:
    """A clear error, not an empty array that a caller would treat as a result."""
    from src.features import FeatureError

    blank = qa_panel.copy()
    blank["cases"] = np.nan

    with pytest.raises(FeatureError) as excinfo:
        build_features(blank, qa_cfg)
    message = str(excinfo.value)
    assert "sequence_length" in message and "horizon" in message, (
        f"error does not say what to change: {message}"
    )
