"""The seasonal forecaster: that it is seasonal, and that it degrades honestly.

Two properties carry this model. It must genuinely respond to the calendar --
otherwise it is an expensive mean -- and it must fail in named ways rather than
silently, because it is the model that answers the months furthest from any data.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from src.config import Config
from src.models.seasonal import (
    SeasonalError,
    months_from_cyclic,
    project_seasonal,
)
from src.panel import complete_index


@pytest.fixture
def seasonal_cfg(cfg: Config) -> Config:
    """Twelve years of monthly history, enough for a ten-year window."""
    return dataclasses.replace(
        cfg,
        data=dataclasses.replace(
            cfg.data,
            start_date=pd.Timestamp("2010-01-01").date(),
            end_date=pd.Timestamp("2021-12-31").date(),
        ),
    )


@pytest.fixture
def seasonal_panel(seasonal_cfg: Config) -> pd.DataFrame:
    """A panel with a strong, unambiguous annual cycle."""
    index = complete_index(seasonal_cfg)
    month = index.get_level_values("date").month.to_numpy()
    wave = np.sin(2 * np.pi * (month - 6) / 12)
    return pd.DataFrame(
        {
            "cases": np.abs(100 + 80 * wave),
            "rainfall": np.abs(120 + 100 * wave),
            "temperature": 27 + 5 * wave,
            "humidity": 70 + 15 * wave,
            "search_interest": 45 + 25 * wave,
            "population": 3.0e7,
            "population_density": 850.0,
        },
        index=index,
    )


# --------------------------------------------------------------------------- #
# The calendar encoding
# --------------------------------------------------------------------------- #


def test_the_month_survives_its_cyclic_encoding() -> None:
    """The model reads the calendar back out of sine and cosine.

    That inversion is what lets it avoid adding a feature to the shared pipeline,
    so it has to be exact for every position, including the wrap at December.
    """
    period = 12
    positions = np.arange(period)
    angle = 2.0 * np.pi * positions / period

    recovered = months_from_cyclic(np.sin(angle), np.cos(angle), period)
    np.testing.assert_array_equal(recovered, positions)


# --------------------------------------------------------------------------- #
# Is it actually seasonal?
# --------------------------------------------------------------------------- #


def test_shifting_the_target_six_months_changes_the_answer_materially(
    seasonal_panel: pd.DataFrame, seasonal_cfg: Config
) -> None:
    """The test that separates this model from an expensive mean.

    Half a year apart is the furthest two months can be in an annual cycle. If
    the profile were flat, the two answers would agree and the whole model would
    be a constant wearing a calendar.
    """
    last = pd.Timestamp(seasonal_panel.loc["Kerala", "cases"].dropna().index.max())

    peak = project_seasonal(
        seasonal_panel, "Kerala", last + pd.DateOffset(months=9), seasonal_cfg
    )
    trough = project_seasonal(
        seasonal_panel, "Kerala", last + pd.DateOffset(months=3), seasonal_cfg
    )

    assert peak.predicted_cases_per_100k > 2.0 * trough.predicted_cases_per_100k, (
        f"six months apart gave {trough.predicted_cases_per_100k:.4f} and "
        f"{peak.predicted_cases_per_100k:.4f}; the profile is not seasonal"
    )


def test_without_a_trend_the_same_month_next_year_repeats(
    seasonal_panel: pd.DataFrame, seasonal_cfg: Config
) -> None:
    """With the trend off it is pure pattern: it repeats rather than compounding.

    That is the property that lets it reach two years where recursion cannot
    reach seven months -- nothing accumulates.
    """
    flat = dataclasses.replace(
        seasonal_cfg,
        seasonal=dataclasses.replace(seasonal_cfg.seasonal, use_trend=False),
    )
    last = pd.Timestamp(seasonal_panel.loc["Kerala", "cases"].dropna().index.max())

    near = project_seasonal(seasonal_panel, "Kerala", last + pd.DateOffset(months=12), flat)
    far = project_seasonal(seasonal_panel, "Kerala", last + pd.DateOffset(months=24), flat)

    assert near.predicted_cases_per_100k == pytest.approx(far.predicted_cases_per_100k)
    assert far.trend_shift == pytest.approx(0.0)


def test_a_rising_series_projects_higher_the_further_out_it_reaches(
    seasonal_cfg: Config
) -> None:
    """With the trend on, two years out must not equal one year out.

    The test that the trend is wired through rather than merely configured. Built
    on a deliberately rising series, because the shipped synthetic panel is
    almost flat once converted to a rate and log-transformed -- a trend term can
    be perfectly correct there and still move nothing.
    """
    index = complete_index(seasonal_cfg)
    month = index.get_level_values("date").month.to_numpy()
    wave = np.sin(2 * np.pi * (month - 6) / 12)
    per_state = len(index) // len(set(index.get_level_values("state")))
    ramp = np.tile(np.arange(per_state), len(set(index.get_level_values("state"))))

    rising = pd.DataFrame(
        {
            "cases": np.abs((100 + 80 * wave) * 1.02**ramp),
            "rainfall": np.abs(120 + 100 * wave), "temperature": 27 + 5 * wave,
            "humidity": 70 + 15 * wave, "search_interest": 45 + 25 * wave,
            "population": 3.0e7, "population_density": 850.0,
        },
        index=index,
    )
    last = pd.Timestamp(rising.loc["Kerala", "cases"].dropna().index.max())

    near = project_seasonal(rising, "Kerala", last + pd.DateOffset(months=12), seasonal_cfg)
    far = project_seasonal(rising, "Kerala", last + pd.DateOffset(months=24), seasonal_cfg)

    assert far.predicted_cases_per_100k > near.predicted_cases_per_100k
    assert far.trend_shift > 0.0


def test_damping_holds_the_trend_below_a_straight_line(seasonal_cfg: Config) -> None:
    """Damping is the guard against a two-year straight-line extrapolation.

    Undamped, the drift is proportional to how far ahead you ask; damped it
    converges. On a rising series the two must visibly disagree, or the damping
    parameter is decorative.
    """
    from src.models.seasonal import _damped_steps

    assert _damped_steps(24, 1.0) == pytest.approx(24.0)
    assert _damped_steps(24, 0.9) < 10.0
    assert _damped_steps(24, 0.0) == pytest.approx(0.0)
    assert _damped_steps(0, 0.9) == pytest.approx(0.0)
    # Converges rather than growing: asking ten years further out adds under a
    # thousandth of a period of drift, where undamped would add 120.
    assert _damped_steps(240, 0.9) == pytest.approx(_damped_steps(120, 0.9), abs=1e-3)
    assert _damped_steps(240, 0.9) < 9.0


def test_the_interval_does_not_widen_with_distance(
    seasonal_panel: pd.DataFrame, seasonal_cfg: Config
) -> None:
    """The opposite of the recursive path, and deliberately so.

    Recursive uncertainty grows because error compounds. This band is the spread
    of past Septembers, which does not change because September is further away.
    """
    last = pd.Timestamp(seasonal_panel.loc["Kerala", "cases"].dropna().index.max())

    def width(months: int) -> float:
        projection = project_seasonal(
            seasonal_panel, "Kerala", last + pd.DateOffset(months=months), seasonal_cfg
        )
        return projection.upper_cases_per_100k - projection.lower_cases_per_100k

    assert width(12) == pytest.approx(width(24))


# --------------------------------------------------------------------------- #
# Degrading honestly
# --------------------------------------------------------------------------- #


def test_a_constant_series_does_not_divide_by_zero(
    seasonal_panel: pd.DataFrame, seasonal_cfg: Config
) -> None:
    """A state whose level never moves gives a zero-variance anchor.

    Dividing by it would turn a perfectly well-behaved constant series into
    infinities, and the pipeline would report success while every downstream
    number was NaN.
    """
    flat = seasonal_panel.copy()
    flat.loc["Kerala", "cases"] = 100.0

    projection = project_seasonal(
        flat, "Kerala", pd.Timestamp("2022-06-01"), seasonal_cfg
    )
    assert np.isfinite(projection.predicted_cases_per_100k)
    assert np.isfinite(projection.lower_cases_per_100k)
    assert np.isfinite(projection.upper_cases_per_100k)


def test_a_short_history_still_answers_and_reports_how_thin_it_is(
    seasonal_panel: pd.DataFrame, seasonal_cfg: Config
) -> None:
    """Fewer years than the window is not a refusal, but it must be visible.

    Silently profiling three Septembers as though they were ten would present a
    thin estimate with the confidence of a thick one.
    """
    short = seasonal_panel.copy()
    dates = short.index.get_level_values("date")
    keep = dates >= pd.Timestamp("2019-01-01")
    short.loc[(short.index.get_level_values("state") == "Kerala") & ~keep, "cases"] = (
        np.nan
    )

    projection = project_seasonal(
        short, "Kerala", pd.Timestamp("2022-06-01"), seasonal_cfg
    )
    assert np.isfinite(projection.predicted_cases_per_100k)
    assert projection.years_observed <= 3
    assert projection.reliability < 1.0, (
        "a three-year profile reported the same reliability as a ten-year one"
    )


def test_a_state_with_no_history_is_refused_by_name(
    seasonal_panel: pd.DataFrame, seasonal_cfg: Config
) -> None:
    blank = seasonal_panel.copy()
    blank.loc["Odisha", "cases"] = np.nan

    with pytest.raises(SeasonalError, match="Odisha"):
        project_seasonal(blank, "Odisha", pd.Timestamp("2022-06-01"), seasonal_cfg)


def test_an_unknown_state_is_refused_by_name(
    seasonal_panel: pd.DataFrame, seasonal_cfg: Config
) -> None:
    with pytest.raises(SeasonalError, match="Atlantis"):
        project_seasonal(
            seasonal_panel, "Atlantis", pd.Timestamp("2022-06-01"), seasonal_cfg
        )


def test_the_projection_does_not_mutate_the_panel(
    seasonal_panel: pd.DataFrame, seasonal_cfg: Config
) -> None:
    before = seasonal_panel.copy()
    project_seasonal(seasonal_panel, "Kerala", pd.Timestamp("2022-06-01"), seasonal_cfg)
    pd.testing.assert_frame_equal(seasonal_panel, before)


def test_the_band_brackets_the_central_estimate(
    seasonal_panel: pd.DataFrame, seasonal_cfg: Config
) -> None:
    projection = project_seasonal(
        seasonal_panel, "Kerala", pd.Timestamp("2022-09-01"), seasonal_cfg
    )
    assert projection.lower_cases_per_100k <= projection.predicted_cases_per_100k
    assert projection.predicted_cases_per_100k <= projection.upper_cases_per_100k
    assert projection.lower_cases_per_100k >= 0.0


def test_turning_the_anchor_off_gives_pure_climatology(
    seasonal_panel: pd.DataFrame, seasonal_cfg: Config
) -> None:
    """The config switch must actually reach the answer."""
    plain = dataclasses.replace(
        seasonal_cfg,
        seasonal=dataclasses.replace(seasonal_cfg.seasonal, use_level_anchor=False),
    )
    anchored = project_seasonal(
        seasonal_panel, "Kerala", pd.Timestamp("2022-06-01"), seasonal_cfg
    )
    unanchored = project_seasonal(
        seasonal_panel, "Kerala", pd.Timestamp("2022-06-01"), plain
    )
    assert anchored.anchored and not unanchored.anchored


# --------------------------------------------------------------------------- #
# Climatology: answering a month years away
# --------------------------------------------------------------------------- #


def test_a_far_future_month_drops_the_anchor_and_the_trend(
    seasonal_panel: pd.DataFrame, seasonal_cfg: Config
) -> None:
    """The degradation that makes a 2030 answer honest.

    How the current year is running says nothing about a month seven years out,
    and a trend extrapolated that far is arithmetic rather than evidence. Both
    are dropped **by the model**, not by the caller remembering to ask correctly.
    """
    last = pd.Timestamp(seasonal_panel.loc["Kerala", "cases"].dropna().index.max())
    far = project_seasonal(
        seasonal_panel, "Kerala", last + pd.DateOffset(years=7), seasonal_cfg
    )

    assert not far.anchored
    assert far.trend_shift == pytest.approx(0.0)


def test_the_same_month_far_out_is_the_same_answer_every_year(
    seasonal_panel: pd.DataFrame, seasonal_cfg: Config
) -> None:
    """August 2030 and August 2031 are the same claim, and must not differ.

    Because the claim is about Augusts, not about the year. Any difference would
    be the model implying it knows something about 2031 that it does not.
    """
    a = project_seasonal(seasonal_panel, "Kerala", pd.Timestamp("2030-08-01"), seasonal_cfg)
    b = project_seasonal(seasonal_panel, "Kerala", pd.Timestamp("2031-08-01"), seasonal_cfg)

    assert a.predicted_cases_per_100k == pytest.approx(b.predicted_cases_per_100k)
    assert a.lower_cases_per_100k == pytest.approx(b.lower_cases_per_100k)


def test_the_typical_year_covers_every_month_with_a_band(
    seasonal_panel: pd.DataFrame, seasonal_cfg: Config
) -> None:
    from src.models.seasonal import climatology_year

    profile = climatology_year(seasonal_panel, "Kerala", seasonal_cfg)

    assert len(profile) == seasonal_cfg.project.seasonal_period
    assert list(profile.columns) == [
        "position", "label", "predicted", "lower", "upper", "observed",
    ]
    assert (profile["lower"] <= profile["predicted"]).all()
    assert (profile["predicted"] <= profile["upper"]).all()
    assert (profile["lower"] >= 0.0).all()
    assert profile["predicted"].max() > 2.0 * profile["predicted"].min(), (
        "the typical year is flat; this profile has no seasonality in it"
    )


def test_the_typical_year_matches_what_a_far_projection_reports(
    seasonal_panel: pd.DataFrame, seasonal_cfg: Config
) -> None:
    """The chart and the headline number must be the same claim.

    They are computed by different functions, so nothing but a test stops them
    drifting into disagreeing about the same month on the same screen.
    """
    from src.models.seasonal import climatology_year

    profile = climatology_year(seasonal_panel, "Kerala", seasonal_cfg)
    august = project_seasonal(
        seasonal_panel, "Kerala", pd.Timestamp("2030-08-01"), seasonal_cfg
    )

    assert profile.loc[7, "predicted"] == pytest.approx(
        august.predicted_cases_per_100k
    )
