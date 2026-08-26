"""Panel fusion, the data-quality report, and the preprocessing pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import Config
from src.panel import (
    PanelError,
    complete_index,
    data_quality_report,
    longest_run,
    mad_outliers,
    summarise_panel,
)
from src.preprocess import (
    FLAG_INTERPOLATED,
    FLAG_MISSING,
    FLAG_OK,
    FLAG_OUTLIER,
    QUALITY_COLUMN,
    detect_outliers,
    find_long_gaps,
    interpolate_short_gaps,
    preprocess,
    reindex_complete,
)

# --------------------------------------------------------------------------- #
# Quality primitives
# --------------------------------------------------------------------------- #


def test_longest_run_counts_consecutive_not_total() -> None:
    """Twelve scattered gaps are interpolable; twelve consecutive are a lost year."""
    scattered = pd.Series([True, False, True, False, True, False])
    consecutive = pd.Series([False, True, True, True, False, False])
    assert longest_run(scattered) == 1
    assert longest_run(consecutive) == 3
    assert longest_run(pd.Series([False, False])) == 0


def test_mad_outliers_catches_a_spike_that_a_z_score_would_hide() -> None:
    """A large outbreak inflates the standard deviation enough to mask itself."""
    values = pd.Series([10.0] * 20 + [500.0])
    flags = mad_outliers(values, threshold=5.0)
    assert flags.iloc[-1]
    assert not flags.iloc[:-1].any()

    z_score = (values - values.mean()).abs() / values.std()
    assert z_score.iloc[-1] < 5.0, "the spike is invisible to a conventional z-score"


def test_mad_outliers_on_a_constant_series_flags_nothing() -> None:
    """Zero MAD must not divide by zero and flag every point."""
    assert not mad_outliers(pd.Series([7.0] * 10), threshold=5.0).any()


def test_mad_outliers_ignores_missing_values() -> None:
    """NaN is never an outlier, and its presence must not defeat the detection."""
    values = pd.Series([1.0] * 10 + [np.nan, 99.0])
    flags = mad_outliers(values, threshold=5.0)
    assert not flags.iloc[10]
    assert flags.iloc[11]


def test_mad_outliers_survives_a_tied_majority() -> None:
    """A state reporting the same low count most months drives the MAD to zero.

    Without the mean-absolute-deviation fallback the scale would be zero and the
    outbreak — the one value that matters — would go unflagged.
    """
    values = pd.Series([0.0] * 30 + [250.0])
    assert (values - values.median()).abs().median() == 0.0, "MAD is degenerate here"
    assert mad_outliers(values, threshold=5.0).iloc[-1]


def test_mad_outliers_on_an_all_missing_series_flags_nothing() -> None:
    assert not mad_outliers(pd.Series([np.nan] * 5), threshold=5.0).any()


# --------------------------------------------------------------------------- #
# Panel shape and reporting
# --------------------------------------------------------------------------- #


def test_complete_index_covers_every_state_and_period(cfg: Config) -> None:
    index = complete_index(cfg)
    assert index.names == ["state", "date"]
    assert len(index) == 3 * 36  # 3 states x 36 months of 2015-2017


def test_report_counts_coverage_and_gaps_per_state_and_variable(
    cfg: Config, panel: pd.DataFrame
) -> None:
    holed = panel.copy()
    holed.loc[("Kerala", slice("2016-03-01", "2016-07-01")), "cases"] = np.nan

    report = data_quality_report(holed, cfg)
    row = report.query("state == 'Kerala' and variable == 'cases'").iloc[0]

    assert row["n_expected"] == 36
    assert row["n_missing"] == 5
    assert row["longest_missing_run"] == 5
    assert row["coverage"] == pytest.approx(31 / 36)
    assert row["first_valid"] == pd.Timestamp("2015-01-01")

    untouched = report.query("state == 'Odisha' and variable == 'cases'").iloc[0]
    assert untouched["n_missing"] == 0


def test_report_handles_a_completely_absent_series(cfg: Config, panel: pd.DataFrame) -> None:
    """An empty state must report zero coverage, not crash on an empty min()."""
    empty = panel.copy()
    empty.loc[("Odisha", slice(None)), "rainfall"] = np.nan
    row = data_quality_report(empty, cfg).query(
        "state == 'Odisha' and variable == 'rainfall'"
    ).iloc[0]
    assert row["coverage"] == 0.0
    assert pd.isna(row["first_valid"])


def test_summary_reports_the_facts_the_modelling_plan_depends_on(
    cfg: Config, panel: pd.DataFrame
) -> None:
    summary = summarise_panel(panel, cfg)
    assert summary.granularity == "monthly"
    assert summary.n_states == 3
    assert summary.periods_per_state == 36
    assert summary.overall_coverage == 1.0
    # 36 monthly points is far below the per-state threshold, so pooling is forced.
    assert summary.pooling_required is True
    assert "REQUIRED" in summary.describe()


def test_summary_flags_states_broken_by_a_boundary_change(cfg: Config) -> None:
    """Andhra Pradesh's series is not comparable across the 2014 Telangana split."""
    import dataclasses

    from src.panel import complete_index as build_index

    wider = dataclasses.replace(
        cfg,
        data=dataclasses.replace(
            cfg.data,
            states=("Andhra Pradesh", "Kerala"),
            start_date=pd.Timestamp("2013-01-01").date(),
        ),
    )
    index = build_index(wider)
    frame = pd.DataFrame({"cases": np.ones(len(index))}, index=index)
    assert summarise_panel(frame, wider).boundary_change_states == ("Andhra Pradesh",)


def test_report_rejects_a_panel_that_is_not_state_date_indexed(panel: pd.DataFrame) -> None:
    with pytest.raises(PanelError, match="must be indexed by"):
        data_quality_report(panel.reset_index())


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #


def test_reindexing_turns_an_absent_period_into_a_visible_nan(
    cfg: Config, panel: pd.DataFrame
) -> None:
    """A row that was simply not there cannot be counted; a NaN can."""
    dropped = panel.drop(index=("Kerala", pd.Timestamp("2016-05-01")))
    restored = reindex_complete(dropped, cfg)
    assert len(restored) == len(panel)
    assert restored.loc[("Kerala", pd.Timestamp("2016-05-01"))].isna().all()


def test_short_climate_gaps_are_filled(cfg: Config, panel: pd.DataFrame) -> None:
    holed = panel.copy()
    holed.loc[("Kerala", pd.Timestamp("2016-05-01")), "rainfall"] = np.nan

    filled, was_interpolated = interpolate_short_gaps(holed, cfg)
    assert not np.isnan(filled.loc[("Kerala", pd.Timestamp("2016-05-01")), "rainfall"])
    assert was_interpolated.loc[("Kerala", pd.Timestamp("2016-05-01")), "rainfall"]
    assert int(was_interpolated.to_numpy().sum()) == 1


def test_long_gaps_are_left_missing_rather_than_invented(
    cfg: Config, panel: pd.DataFrame
) -> None:
    """max_interpolation_gap is 2; a five-month hole must survive as NaN."""
    holed = panel.copy()
    holed.loc[("Kerala", slice("2016-03-01", "2016-07-01")), "rainfall"] = np.nan

    filled, _ = interpolate_short_gaps(holed, cfg)
    assert filled.loc[("Kerala", slice("2016-03-01", "2016-07-01")), "rainfall"].isna().all()

    gaps = find_long_gaps(holed, cfg)
    row = gaps.query("state == 'Kerala' and variable == 'rainfall'").iloc[0]
    assert row["gap_length"] == 5


def test_a_long_gap_is_not_partially_filled(cfg: Config, panel: pd.DataFrame) -> None:
    """Pandas' limit= would invent the first two periods of a five-period hole.

    Partial filling is worse than either alternative: the series looks continuous
    where it is not, and the invented values sit adjacent to the real gap.
    """
    holed = panel.copy()
    holed.loc[("Kerala", slice("2016-03-01", "2016-07-01")), "temperature"] = np.nan

    filled, was_interpolated = interpolate_short_gaps(holed, cfg)
    window = filled.loc[("Kerala", slice("2016-03-01", "2016-07-01")), "temperature"]
    assert window.isna().sum() == 5, "no period of an over-long run may be filled"
    assert not was_interpolated["temperature"].any()


def test_case_counts_are_never_interpolated(cfg: Config, panel: pd.DataFrame) -> None:
    """A month without surveillance is not a month with an estimable case count."""
    holed = panel.copy()
    holed.loc[("Kerala", pd.Timestamp("2016-05-01")), "cases"] = np.nan

    filled, was_interpolated = interpolate_short_gaps(holed, cfg)
    assert np.isnan(filled.loc[("Kerala", pd.Timestamp("2016-05-01")), "cases"])
    assert not was_interpolated["cases"].any()


def test_interpolation_never_crosses_a_state_boundary(cfg: Config, panel: pd.DataFrame) -> None:
    """Kerala's rainfall must never be filled from Odisha's."""
    holed = panel.copy()
    # Blank the whole of Kerala's rainfall except a single early observation, so any
    # fill after it could only have come from the neighbouring state's block.
    holed.loc[("Kerala", slice("2015-02-01", None)), "rainfall"] = np.nan

    filled, _ = interpolate_short_gaps(holed, cfg)
    assert filled.loc[("Kerala", slice("2015-02-01", None)), "rainfall"].isna().all()


def test_leading_gaps_are_not_back_filled(cfg: Config, panel: pd.DataFrame) -> None:
    """Filling the start of a series extrapolates backwards from a future value."""
    holed = panel.copy()
    holed.loc[("Kerala", slice(None, "2015-02-01")), "rainfall"] = np.nan

    filled, _ = interpolate_short_gaps(holed, cfg)
    assert filled.loc[("Kerala", slice(None, "2015-02-01")), "rainfall"].isna().all()


def test_outliers_are_flagged_and_kept(cfg: Config, panel: pd.DataFrame) -> None:
    """The extremes are the outbreaks — the events the model exists to predict."""
    spiked = panel.copy()
    spiked.loc[("Odisha", pd.Timestamp("2016-09-01")), "cases"] = 5000.0

    flags = detect_outliers(spiked, cfg)
    assert flags.loc[("Odisha", pd.Timestamp("2016-09-01")), "cases"]

    result = preprocess(spiked, cfg)
    assert result.panel.loc[("Odisha", pd.Timestamp("2016-09-01")), "cases"] == 5000.0
    assert result.n_outliers >= 1


def test_outliers_are_detected_per_state_not_across_the_panel(cfg: Config) -> None:
    """A state with a higher baseline must not flag its whole series as extreme."""
    index = complete_index(cfg)
    values = pd.Series(1.0, index=index)
    values.loc[("Tamil Nadu", slice(None))] = 1000.0
    frame = pd.DataFrame({"cases": values})

    assert not detect_outliers(frame, cfg)["cases"].any()


def test_quality_flag_records_the_most_severe_condition(
    cfg: Config, panel: pd.DataFrame
) -> None:
    dirty = panel.copy()
    dirty.loc[("Kerala", pd.Timestamp("2016-05-01")), "rainfall"] = np.nan   # short -> filled
    dirty.loc[("Kerala", slice("2017-01-01", "2017-06-01")), "cases"] = np.nan  # stays missing
    dirty.loc[("Odisha", pd.Timestamp("2016-09-01")), "cases"] = 5000.0      # outlier

    result = preprocess(dirty, cfg)
    quality = result.panel[QUALITY_COLUMN]

    assert quality.loc[("Kerala", pd.Timestamp("2016-05-01"))] == FLAG_INTERPOLATED
    assert quality.loc[("Kerala", pd.Timestamp("2017-03-01"))] == FLAG_MISSING
    assert quality.loc[("Odisha", pd.Timestamp("2016-09-01"))] == FLAG_OUTLIER
    assert quality.loc[("Tamil Nadu", pd.Timestamp("2015-06-01"))] == FLAG_OK


def test_preprocess_reports_what_it_did(cfg: Config, panel: pd.DataFrame) -> None:
    dirty = panel.copy()
    dirty.loc[("Kerala", pd.Timestamp("2016-05-01")), "rainfall"] = np.nan
    dirty.loc[("Kerala", slice("2017-01-01", "2017-06-01")), "cases"] = np.nan

    result = preprocess(dirty, cfg)
    assert result.n_interpolated == 1
    assert result.n_still_missing == 6
    assert "Interpolated" in result.describe()
    assert not result.long_gaps.empty


def test_preprocess_is_idempotent(cfg: Config, panel: pd.DataFrame) -> None:
    """Running it twice must not fill more, or re-flag differently."""
    first = preprocess(panel, cfg)
    columns = list(panel.columns)
    second = preprocess(first.panel[columns], cfg)
    assert second.n_interpolated == 0
    pd.testing.assert_frame_equal(first.panel[columns], second.panel[columns])


def test_preprocess_fits_nothing() -> None:
    """The review gate, enforced: no estimator may be constructed in this module."""
    import inspect

    import src.preprocess as module

    source = inspect.getsource(module)
    body = source.split('"""', 2)[-1]  # skip the module docstring, which discusses them
    for forbidden in (".fit(", ".fit_transform(", "Scaler(", "Imputer("):
        assert forbidden not in body, f"{forbidden} appeared in src/preprocess.py"