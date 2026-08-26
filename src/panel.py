"""Fusion and data-quality reporting.

:func:`assemble_panel` concatenates every configured source's long rows and pivots
them to a wide panel indexed by ``(state, date)``. The index is a complete
``states x periods`` grid, so a period no source reported becomes an explicit NaN
rather than a silently absent row. A gap you can see is a gap you can decide about.

:func:`data_quality_report` is the companion: coverage, missing runs, outliers and
valid date ranges per ``(state, variable)``. It answers the questions that decide
the modelling plan — is the series monthly or weekly, how many time points does
each state actually have, and is that enough to fit anything per-state.

Nothing here imputes, scales or fits.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from src.config import Config, load_config
from src.sources import PANEL_KEYS
from src.sources.base import GRANULARITY_FREQ, sources_for
from src.sources.registry import affected_by_boundary_changes, normalise_state

#: Scale factor making the median absolute deviation comparable to a standard
#: deviation for normally distributed data.
MAD_TO_SIGMA = 1.4826

#: The same for the mean absolute deviation, used when the MAD degenerates to zero.
MEAN_AD_TO_SIGMA = 1.2533


class PanelError(RuntimeError):
    """Raised when sources cannot be fused into a well-formed panel."""


@dataclass(frozen=True)
class PanelSummary:
    """Headline facts about an assembled panel.

    These are the numbers the modelling plan depends on, which is why they are a
    typed object rather than a printed table: ``periods_per_state`` decides
    whether a per-state model is viable at all.
    """

    granularity: str
    n_states: int
    n_periods: int
    periods_per_state: int
    first_date: date
    last_date: date
    variables: tuple[str, ...]
    overall_coverage: float
    states_below_coverage: tuple[str, ...]
    boundary_change_states: tuple[str, ...]
    pooling_required: bool

    def describe(self) -> str:
        """A short human-readable verdict, suitable for the report and the log."""
        lines = [
            f"Granularity      : {self.granularity}",
            f"States           : {self.n_states}",
            f"Periods / state  : {self.periods_per_state} "
            f"({self.first_date} to {self.last_date})",
            f"Variables        : {', '.join(self.variables)}",
            f"Overall coverage : {self.overall_coverage:.1%}",
        ]
        if self.states_below_coverage:
            lines.append(f"Low coverage     : {', '.join(self.states_below_coverage)}")
        if self.boundary_change_states:
            lines.append(
                f"Boundary breaks  : {', '.join(self.boundary_change_states)} "
                "(state redefined mid-window; series not comparable across the break)"
            )
        lines.append(
            "Pooling          : "
            + (
                "REQUIRED - too few periods per state for a per-state model"
                if self.pooling_required
                else "optional at this series length"
            )
        )
        return "\n".join(lines)


def assemble_panel(cfg: Config | None = None) -> pd.DataFrame:
    """Fetch every configured source and fuse it into one wide panel.

    Args:
        cfg: Loaded configuration; read from ``config.yaml`` when omitted.

    Returns:
        A frame indexed by ``(state, date)`` with one column per variable, spanning
        the complete grid of configured states and periods. Missing observations
        are NaN.

    Raises:
        PanelError: no source produced rows, or two sources claim the same variable.
    """
    cfg = cfg or load_config()
    sources = sources_for(cfg)

    frames: list[pd.DataFrame] = []
    owners: dict[str, str] = {}
    for source in sources:
        frame = source.fetch(cfg)
        for variable in frame["variable"].unique():
            existing = owners.get(variable)
            if existing is not None:
                raise PanelError(
                    f"variable {variable!r} is emitted by both {existing!r} and {source.name!r}; "
                    "variable names must be unique across sources"
                )
            owners[variable] = source.name
        frames.append(frame)

    if not frames or all(frame.empty for frame in frames):
        raise PanelError(
            f"no rows returned by sources {[source.name for source in sources]} "
            f"for {cfg.data.start_date}..{cfg.data.end_date}"
        )

    long = pd.concat(frames, ignore_index=True)
    wide = long.pivot_table(
        index=list(PANEL_KEYS), columns="variable", values="value", aggfunc="first"
    )
    wide.columns.name = None
    return wide.reindex(complete_index(cfg)).sort_index()


def load_panel(cfg: Config, synthetic: bool) -> pd.DataFrame:
    """Assemble the real panel, or generate the stand-in.

    The one place the choice is made, called by every script. It used to be a
    four-line ``if`` repeated in seven of them, each reaching into
    ``scripts/run_baselines.py`` for the generator -- which made a CLI entry point
    an import target and meant the scripts could only run with the project root
    already on ``sys.path``.

    Args:
        cfg: Loaded configuration.
        synthetic: Generate a stand-in instead of reading ``data/``. Results
            produced this way describe a generator, not dengue; see
            :mod:`src.synthetic`.
    """
    if synthetic:
        # Imported here rather than at module scope: src.synthetic imports
        # complete_index from this module, and a top-level import would close
        # the cycle.
        from src.synthetic import synthetic_panel

        return synthetic_panel(cfg)
    return assemble_panel(cfg)


def complete_index(cfg: Config) -> pd.MultiIndex:
    """The full ``(state, date)`` grid the panel must span.

    Building the panel against this rather than against whatever the sources
    happened to return is what turns an absent period into a visible NaN.
    """
    states = sorted({normalise_state(state) for state in cfg.data.states})
    return pd.MultiIndex.from_product(
        [states, period_index(cfg)], names=list(PANEL_KEYS)
    )


def period_index(cfg: Config) -> pd.DatetimeIndex:
    """Regular period-start index over the configured window."""
    try:
        freq = GRANULARITY_FREQ[cfg.project.granularity]
    except KeyError:
        raise PanelError(f"unsupported granularity {cfg.project.granularity!r}") from None
    return pd.date_range(cfg.data.start_date, cfg.data.end_date, freq=freq)


# --------------------------------------------------------------------------- #
# Data quality
# --------------------------------------------------------------------------- #


def data_quality_report(panel: pd.DataFrame, cfg: Config | None = None) -> pd.DataFrame:
    """Per ``(state, variable)`` coverage, gaps, outliers and valid range.

    Args:
        panel: Wide panel from :func:`assemble_panel`.
        cfg: Loaded configuration; read from ``config.yaml`` when omitted.

    Returns:
        One row per state and variable, with ``n_expected``, ``n_present``,
        ``coverage``, ``n_missing``, ``longest_missing_run``, ``first_valid``,
        ``last_valid``, ``n_outliers`` and ``pct_outliers``.

    Note:
        Outliers are counted using a median/MAD rule rather than mean/standard
        deviation. Dengue counts are spiky by nature, and an outbreak inflates the
        standard deviation enough to hide itself from a z-score.
    """
    cfg = cfg or load_config()
    _require_panel_shape(panel)
    threshold = cfg.quality.outlier_mad_threshold

    rows: list[dict[str, object]] = []
    for state, group in panel.groupby(level="state", sort=True):
        series_by_variable = group.droplevel("state")
        for variable in panel.columns:
            series = series_by_variable[variable]
            present = series.notna()
            valid = series.dropna()
            rows.append(
                {
                    "state": state,
                    "variable": variable,
                    "n_expected": int(len(series)),
                    "n_present": int(present.sum()),
                    "coverage": float(present.mean()) if len(series) else 0.0,
                    "n_missing": int((~present).sum()),
                    "longest_missing_run": longest_run(~present),
                    "first_valid": valid.index.min() if not valid.empty else pd.NaT,
                    "last_valid": valid.index.max() if not valid.empty else pd.NaT,
                    "n_outliers": int(mad_outliers(series, threshold).sum()),
                }
            )

    report = pd.DataFrame(rows)
    report["pct_outliers"] = np.where(
        report["n_present"] > 0, report["n_outliers"] / report["n_present"], 0.0
    )
    return report.sort_values(["variable", "state"]).reset_index(drop=True)


def summarise_panel(panel: pd.DataFrame, cfg: Config | None = None) -> PanelSummary:
    """Reduce a panel to the facts that decide the modelling plan."""
    cfg = cfg or load_config()
    _require_panel_shape(panel)

    dates = panel.index.get_level_values("date")
    states = panel.index.get_level_values("state").unique()
    report = data_quality_report(panel, cfg)

    per_state_coverage = report.groupby("state")["coverage"].mean()
    low = per_state_coverage[per_state_coverage < cfg.quality.min_coverage]
    periods_per_state = int(len(dates.unique()))

    return PanelSummary(
        granularity=cfg.project.granularity,
        n_states=int(len(states)),
        n_periods=int(len(panel)),
        periods_per_state=periods_per_state,
        first_date=dates.min().date(),
        last_date=dates.max().date(),
        variables=tuple(panel.columns),
        overall_coverage=float(panel.notna().to_numpy().mean()),
        states_below_coverage=tuple(sorted(low.index)),
        boundary_change_states=tuple(
            sorted(
                affected_by_boundary_changes(
                    states, cfg.data.start_date, cfg.data.end_date
                )
            )
        ),
        pooling_required=periods_per_state < cfg.quality.min_periods_for_per_state_model,
    )


def longest_run(flags: pd.Series) -> int:
    """Length of the longest consecutive run of True values.

    A run matters more than a count: twelve scattered missing months can be
    interpolated, whereas twelve consecutive ones are an absent year.
    """
    values = np.asarray(flags, dtype=bool)
    if not values.any():
        return 0
    longest = current = 0
    for flag in values:
        current = current + 1 if flag else 0
        longest = max(longest, current)
    return longest


def mad_outliers(series: pd.Series, threshold: float) -> pd.Series:
    """Flag values further than ``threshold`` robust sigmas from the median.

    Missing values are never flagged. A genuinely constant series yields no flags
    rather than flagging everything.

    Note:
        The MAD collapses to zero whenever more than half the observations are
        identical — which is common in dengue counts, where a state can report the
        same low number, or zero, for most of the year. A plain MAD rule would then
        divide by zero and silently flag nothing, hiding exactly the outbreak it was
        meant to catch. When that happens the scale falls back to the mean absolute
        deviation, which stays positive as long as any two values differ.
    """
    values = series.astype(float)
    if values.notna().sum() == 0:
        return pd.Series(False, index=series.index)

    deviation = (values - values.median()).abs()
    scale = MAD_TO_SIGMA * deviation.median()
    if not np.isfinite(scale) or scale == 0:
        scale = MEAN_AD_TO_SIGMA * deviation.mean()
    if not np.isfinite(scale) or scale == 0:
        return pd.Series(False, index=series.index)

    return (deviation / scale > threshold).fillna(False)


def _require_panel_shape(panel: pd.DataFrame) -> None:
    """Reject anything that is not a ``(state, date)``-indexed wide panel."""
    if not isinstance(panel.index, pd.MultiIndex) or tuple(panel.index.names) != PANEL_KEYS:
        raise PanelError(
            f"panel must be indexed by {list(PANEL_KEYS)}, got {list(panel.index.names)}"
        )
    if panel.index.duplicated().any():
        raise PanelError("panel index contains duplicate (state, date) pairs")