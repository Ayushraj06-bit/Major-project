"""Panel cleaning, with every leakage-prone operation deferred.

**Nothing in this module fits anything to the data.** No scaler, no imputer
learned from the series, no PCA, no target encoding, no ``.fit()`` of any kind.
If you are here to add a ``MinMaxScaler`` or a ``StandardScaler``, stop: scaling
belongs inside the cross-validation fold loop in :mod:`src.evaluate`, fitted on
each fold's training portion alone. Fitting it here would fit it to the test
period too, which inflates every metric in the report and is the single most
common — and most easily spotted — bug in a forecasting project.

What is safe here is anything decided *per row from its own past*, or from fixed
constants in config: reindexing to a regular grid, short-gap interpolation within
a state, and flagging. These need no parameters estimated across the train/test
boundary.

The gap policy is deliberately narrow. Climate is a smooth physical field, so a
one- or two-period hole can be interpolated defensibly. **Case counts are never
interpolated.** A month with no dengue surveillance report is not a month with an
estimable number of cases — filling it invents an epidemic curve and then trains a
model to reproduce the invention. Those gaps stay NaN and are recorded in
``data_quality``.

Every transformation is a standalone function; :func:`preprocess` composes them in
order and is the only entry point callers need.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import Config, load_config
from src.panel import PanelError, complete_index, mad_outliers
from src.sources import PANEL_KEYS

#: Column added by :func:`flag_quality`, summarising each row's provenance.
QUALITY_COLUMN = "data_quality"

#: Per-row quality flags, most severe last.
FLAG_OK = "ok"
FLAG_INTERPOLATED = "interpolated"
FLAG_OUTLIER = "outlier"
FLAG_MISSING = "missing"

#: Suffix of the per-variable boolean columns marking interpolated observations.
INTERPOLATED_SUFFIX = "_interpolated"


@dataclass(frozen=True)
class PreprocessResult:
    """A cleaned panel and an account of what was done to it.

    The report is returned rather than logged so that the count of interpolated
    and still-missing observations can be cited in the write-up instead of
    recalled.
    """

    panel: pd.DataFrame
    n_interpolated: int
    n_still_missing: int
    n_outliers: int
    long_gaps: pd.DataFrame

    def describe(self) -> str:
        """A short human-readable account, suitable for the log and the report."""
        lines = [
            f"Interpolated   : {self.n_interpolated} observation(s)",
            f"Left missing   : {self.n_still_missing} observation(s)",
            f"Outliers flagged: {self.n_outliers} observation(s) (flagged, not removed)",
        ]
        if not self.long_gaps.empty:
            worst = self.long_gaps.nlargest(3, "gap_length")
            described = ", ".join(
                f"{row.state}/{row.variable} ({row.gap_length})" for row in worst.itertuples()
            )
            lines.append(
                f"Long gaps      : {len(self.long_gaps)} exceeded the interpolation "
                f"limit and were left missing — worst: {described}"
            )
        return "\n".join(lines)


def preprocess(panel: pd.DataFrame, cfg: Config | None = None) -> PreprocessResult:
    """Clean an assembled panel, composing the transformations in order.

    The order matters. Reindexing first makes every gap explicit, so interpolation
    and gap measurement see the same complete grid. Outliers are detected on the
    observed values before interpolation adds any synthetic ones, so a flagged
    outlier is always a real observation.

    Args:
        panel: Wide panel from :func:`src.panel.assemble_panel`.
        cfg: Loaded configuration; read from ``config.yaml`` when omitted.

    Returns:
        The cleaned panel plus counts of what changed.

    Raises:
        PanelError: the input is not a ``(state, date)``-indexed wide panel.
    """
    cfg = cfg or load_config()
    _require_panel_shape(panel)

    reindexed = reindex_complete(panel, cfg)
    outliers = detect_outliers(reindexed, cfg)
    long_gaps = find_long_gaps(reindexed, cfg)
    interpolated, was_interpolated = interpolate_short_gaps(reindexed, cfg)
    flagged = flag_quality(interpolated, was_interpolated, outliers)

    return PreprocessResult(
        panel=flagged,
        n_interpolated=int(was_interpolated.to_numpy().sum()),
        n_still_missing=int(interpolated[list(reindexed.columns)].isna().to_numpy().sum()),
        n_outliers=int(outliers.to_numpy().sum()),
        long_gaps=long_gaps,
    )


# --------------------------------------------------------------------------- #
# One function per transformation
# --------------------------------------------------------------------------- #


def reindex_complete(panel: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Place the panel on the complete ``(state, date)`` grid.

    Periods no source reported become explicit NaN rows. Every later step then
    operates on a regular index, and a gap is something you can count rather than
    something that is simply not there.
    """
    return panel.reindex(complete_index(cfg)).sort_index()


def interpolate_short_gaps(
    panel: pd.DataFrame, cfg: Config
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fill gaps of at most ``max_interpolation_gap`` periods, within each state.

    Only the variables named in ``preprocess.interpolate_variables`` are touched —
    climate by default. Interpolation runs **per state**, never across the panel,
    so Kerala's rainfall can never be filled from Rajasthan's.

    Leading and trailing gaps are never filled. ``limit_area="inside"`` confines
    interpolation to holes bounded by real observations on both sides; filling the
    start of a series would extrapolate backwards from a single future value.

    Note:
        A gap is filled only if the **whole run** is short enough. pandas'
        ``limit=`` argument does not express that: it caps how many consecutive
        values are filled, so a five-period hole under ``limit=2`` comes back with
        its first two periods invented and the rest missing — the worst of both
        options. Runs are therefore measured first and long ones masked back out.

    Args:
        panel: Reindexed panel.
        cfg: Loaded configuration.

    Returns:
        The panel with short gaps filled, and a boolean frame of the same shape
        marking which cells were synthesised.
    """
    targets = [
        variable for variable in cfg.preprocess.interpolate_variables if variable in panel.columns
    ]
    was_missing = panel.isna()
    filled = panel.copy()
    limit = cfg.preprocess.max_interpolation_gap
    method = cfg.preprocess.interpolation_method

    # Grouped by state so a gap in one state is never filled from another's values.
    for _, group in panel.groupby(level="state", sort=False):
        dated = group.droplevel("state")
        for variable in targets:
            filled.loc[group.index, variable] = _fill_short_runs(
                dated[variable], method=method, limit=limit
            ).to_numpy()

    was_interpolated = was_missing & filled.notna()
    return filled, was_interpolated


def _fill_short_runs(series: pd.Series, method: str, limit: int) -> pd.Series:
    """Interpolate one series, then undo any fill inside an over-long run."""
    interpolated = series.interpolate(method=method, limit_area="inside")
    for start, end, length in _missing_runs(series):
        if length > limit:
            interpolated.loc[start:end] = np.nan
    return interpolated


def find_long_gaps(panel: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """List runs of missing values too long to interpolate.

    These are the gaps that stay NaN. Reporting them is the point: a nine-month
    hole in a state's case series is a fact about surveillance, and the decision to
    drop that state, shorten the window, or model around it should be made
    deliberately rather than papered over.

    Returns:
        One row per gap, with ``state``, ``variable``, ``start``, ``end`` and
        ``gap_length``. Empty when every gap is short.
    """
    limit = cfg.preprocess.max_interpolation_gap
    rows: list[dict[str, object]] = []

    for state, group in panel.groupby(level="state", sort=True):
        dated = group.droplevel("state")
        for variable in panel.columns:
            for start, end, length in _missing_runs(dated[variable]):
                if length > limit:
                    rows.append(
                        {
                            "state": state,
                            "variable": variable,
                            "start": start,
                            "end": end,
                            "gap_length": length,
                        }
                    )

    columns = ["state", "variable", "start", "end", "gap_length"]
    return pd.DataFrame(rows, columns=columns)


def detect_outliers(panel: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Flag extreme values per state and variable, without removing anything.

    Removal would be wrong here. In a dengue series the extreme values *are* the
    outbreaks — the events the model exists to predict. A flag lets them be
    inspected, excluded from a sensitivity run, or discussed in the report, while
    the model still trains on them.

    Uses a median/MAD rule for the reason given in :func:`src.panel.mad_outliers`:
    a large outbreak inflates the standard deviation enough to hide itself from a
    conventional z-score.

    Returns:
        A boolean frame aligned to ``panel``.
    """
    threshold = cfg.quality.outlier_mad_threshold
    grouped = panel.groupby(level="state", sort=False)
    return pd.DataFrame(
        {
            column: grouped[column].transform(lambda values: mad_outliers(values, threshold))
            for column in panel.columns
        },
        index=panel.index,
    ).astype(bool)


def flag_quality(
    panel: pd.DataFrame, was_interpolated: pd.DataFrame, outliers: pd.DataFrame
) -> pd.DataFrame:
    """Attach provenance columns describing how trustworthy each row is.

    Adds one boolean ``<variable>_interpolated`` column per interpolated variable,
    and a single ``data_quality`` column summarising the row: ``missing`` if any
    value is still absent, else ``outlier`` if any is extreme, else
    ``interpolated`` if any was synthesised, else ``ok``.

    The per-variable columns are what feature engineering and the dashboard need;
    the summary column is what a human reads.
    """
    result = panel.copy()
    variables = list(panel.columns)

    for variable in variables:
        if was_interpolated[variable].any():
            result[f"{variable}{INTERPOLATED_SUFFIX}"] = was_interpolated[variable].to_numpy()

    quality = pd.Series(FLAG_OK, index=panel.index, dtype=object)
    # Assigned least severe first, so a more severe flag overwrites a milder one.
    quality[was_interpolated[variables].any(axis=1).to_numpy()] = FLAG_INTERPOLATED
    quality[outliers[variables].any(axis=1).to_numpy()] = FLAG_OUTLIER
    quality[panel[variables].isna().any(axis=1).to_numpy()] = FLAG_MISSING
    result[QUALITY_COLUMN] = quality
    return result


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

#: The composed pipeline, exposed so the order is inspectable and testable.
PIPELINE: tuple[Callable[..., object], ...] = (
    reindex_complete,
    detect_outliers,
    find_long_gaps,
    interpolate_short_gaps,
    flag_quality,
)


def _missing_runs(series: pd.Series) -> Sequence[tuple[pd.Timestamp, pd.Timestamp, int]]:
    """Contiguous runs of NaN in a date-indexed series, as (start, end, length)."""
    missing = series.isna().to_numpy()
    if not missing.any():
        return []

    runs: list[tuple[pd.Timestamp, pd.Timestamp, int]] = []
    index = series.index
    start: int | None = None
    for position, flag in enumerate(missing):
        if flag and start is None:
            start = position
        elif not flag and start is not None:
            runs.append((index[start], index[position - 1], position - start))
            start = None
    if start is not None:
        runs.append((index[start], index[-1], len(missing) - start))
    return runs


def _require_panel_shape(panel: pd.DataFrame) -> None:
    """Reject anything that is not a ``(state, date)``-indexed wide panel."""
    if not isinstance(panel.index, pd.MultiIndex) or tuple(panel.index.names) != PANEL_KEYS:
        raise PanelError(
            f"panel must be indexed by {list(PANEL_KEYS)}, got {list(panel.index.names)}"
        )
    non_numeric = [
        column
        for column in panel.columns
        if not pd.api.types.is_numeric_dtype(panel[column]) and not _is_bool(panel[column])
    ]
    if non_numeric:
        raise PanelError(f"panel columns must be numeric; {non_numeric} are not")


def _is_bool(series: pd.Series) -> bool:
    """Whether a column is boolean, which numeric checks treat separately."""
    return pd.api.types.is_bool_dtype(series) or series.dtype == np.bool_