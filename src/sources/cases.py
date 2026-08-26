"""Dengue case counts per state.

Provenance is manual. NCVBDC (formerly NVBDCP) publishes state-wise dengue cases
and deaths as PDF and HTML tables; MoSPI and Indiastat republish the same series
in spreadsheet form. None of them offers a stable machine API, so the raw export
is placed in ``data/raw/cases/`` by hand and treated as immutable thereafter.

Two layouts are accepted, because the published tables appear in both:

* **long** — one row per state, date and value;
* **wide** — one row per state, one column per period, which is how the annual
  state-wise tables are laid out.

Whether the underlying series is monthly or weekly is the project's largest open
question. This parser does not assume: it reads what is there, and the
data-quality report states what was actually found.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import Config
from src.sources.base import BaseDataSource, SourceError

#: Column names accepted for each role, matched case-insensitively.
_STATE_COLUMNS = ("state", "state/ut", "state_ut", "states/uts", "state name")
_DATE_COLUMNS = ("date", "month", "period", "week", "year_month", "yearmonth")
_VALUE_COLUMNS = ("value", "cases", "dengue_cases", "count", "cases_reported")

_ACQUISITION = (
    "Download the state-wise dengue case table from NCVBDC "
    "(https://ncvbdc.mohfw.gov.in/index4.php?lang=1&level=0&linkid=431&lid=3715) "
    "or an equivalent MoSPI/Indiastat export, save it as CSV or XLSX, and place it "
    "at the path above. Keep the file exactly as downloaded."
)


class CasesSource(BaseDataSource):
    """Historical dengue case counts, the epidemiological backbone of the panel."""

    name = "cases"
    variables = ("cases",)
    provenance = "NCVBDC / MoSPI / Indiastat state-wise dengue reports"

    #: Any of these filenames is accepted, so the user need not rename their download.
    candidate_filenames: tuple[str, ...] = (
        "cases.csv",
        "cases.xlsx",
        "dengue_cases.csv",
        "dengue_cases.xlsx",
    )

    def fetch_raw(self, cfg: Config) -> Path:
        """Locate the manually-placed case export."""
        directory = self.raw_dir(cfg)
        for filename in self.candidate_filenames:
            candidate = directory / filename
            if candidate.is_file():
                return candidate
        return self.require_raw_file(cfg, self.candidate_filenames[0], _ACQUISITION)

    def parse(self, raw: Path, cfg: Config) -> pd.DataFrame:
        """Read the export in whichever of the two layouts it uses."""
        frame = _read_tabular(raw)
        state_column = _find_column(frame, _STATE_COLUMNS, raw)
        date_column = _match_column(frame, _DATE_COLUMNS)

        if date_column is None:
            long = _melt_wide(frame, state_column=state_column, path=raw)
        else:
            value_column = _find_column(frame, _VALUE_COLUMNS, raw)
            long = frame.loc[:, [state_column, date_column, value_column]].copy()
            long.columns = ["state", "date", "value"]

        long["variable"] = self.variables[0]
        long["value"] = pd.to_numeric(long["value"], errors="coerce")
        # Rows with no state or date cannot be placed in the panel; rows with a
        # missing value are a real observation gap and are kept as NaN.
        long = long.dropna(subset=["state", "date"])
        return long.loc[:, ["state", "date", "variable", "value"]]


def _read_tabular(path: Path) -> pd.DataFrame:
    """Read a CSV or Excel export into a frame, with a usable error if that fails."""
    try:
        if path.suffix.lower() in {".xlsx", ".xls"}:
            return pd.read_excel(path)
        return pd.read_csv(path)
    except ImportError as exc:
        raise SourceError(
            f"reading {path.name} needs an Excel engine: pip install openpyxl ({exc})"
        ) from exc
    except Exception as exc:
        raise SourceError(f"could not read {path}: {exc}") from exc


def _match_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    """First column whose name matches a candidate, ignoring case and spacing."""
    normalised = {str(column).strip().lower(): column for column in frame.columns}
    for candidate in candidates:
        if candidate in normalised:
            return normalised[candidate]
    return None


def _find_column(frame: pd.DataFrame, candidates: tuple[str, ...], path: Path) -> str:
    """Like :func:`_match_column`, but required."""
    column = _match_column(frame, candidates)
    if column is None:
        raise SourceError(
            f"{path.name}: no column matching {list(candidates)}; found {list(frame.columns)}"
        )
    return column


def _melt_wide(frame: pd.DataFrame, state_column: str, path: Path) -> pd.DataFrame:
    """Reshape a one-column-per-period table to long rows.

    Every column other than the state column is treated as a period label. A
    column whose header does not parse as a date is an error rather than a
    silently dropped year — a missing year would shorten the series without
    anyone noticing.
    """
    period_columns = [column for column in frame.columns if column != state_column]
    if not period_columns:
        raise SourceError(f"{path.name}: no period columns beside {state_column!r}")

    unparseable = [
        column
        for column in period_columns
        if pd.to_datetime(str(column), errors="coerce") is pd.NaT
    ]
    if unparseable:
        raise SourceError(
            f"{path.name}: column header(s) {unparseable} do not parse as dates. "
            "Wide exports must use period labels such as '2019-07' or 'Jul 2019'."
        )

    long = frame.melt(id_vars=[state_column], value_vars=period_columns,
                      var_name="date", value_name="value")
    long = long.rename(columns={state_column: "state"})
    return long
