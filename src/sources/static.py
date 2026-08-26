"""Static state attributes: population, population density, adjacency.

Population is the denominator for the modelling target — cases per 100,000 —
which is what makes states comparable in a pooled model, so this source is not
optional even though it carries no temporal signal of its own.

Two things are deliberately handled differently:

* **Population changes slowly but is not constant.** Census figures land a decade
  apart. Rather than freeze one census year across a 14-year window, intercensal
  years are interpolated geometrically between the anchors provided, and any year
  outside them is held flat at the nearest anchor. The interpolation is visible
  here rather than hidden in feature engineering.
* **Adjacency is topology, not a time series.** It has no ``(state, date)`` shape,
  so it does not enter the panel. It is exposed through the canonical registry and
  read directly by the spatial-lag feature builder.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import Config
from src.sources.base import GRANULARITY_FREQ, BaseDataSource, SourceError

_ACQUISITION = (
    "Create a CSV with columns state,year,population,area_sq_km using Census of "
    "India figures (censusindia.gov.in). One row per state per census year is "
    "enough — 2001 and 2011 anchors let intercensal years be interpolated. Area is "
    "constant per state and may be repeated."
)


class StaticSource(BaseDataSource):
    """Population and population density, broadcast onto the panel's time index."""

    name = "demographic"
    variables = ("population", "population_density")
    provenance = "Census of India"

    filename = "population.csv"

    def fetch_raw(self, cfg: Config) -> Path:
        """Locate the manually-prepared census extract."""
        return self.require_raw_file(cfg, self.filename, _ACQUISITION)

    def parse(self, raw: Path, cfg: Config) -> pd.DataFrame:
        """Interpolate census anchors onto the study's time index."""
        table = _read_census(raw)
        index = _time_index(cfg)
        frames = [
            _interpolate_state(state, group, index)
            for state, group in table.groupby("state", sort=True)
        ]
        long = pd.concat(frames, ignore_index=True)
        return long.loc[:, ["state", "date", "variable", "value"]]


def _read_census(path: Path) -> pd.DataFrame:
    """Read and validate the census extract."""
    try:
        table = pd.read_csv(path)
    except Exception as exc:
        raise SourceError(f"could not read {path}: {exc}") from exc

    required = {"state", "year", "population", "area_sq_km"}
    missing = required - {str(column).strip().lower() for column in table.columns}
    if missing:
        raise SourceError(
            f"{path.name}: missing column(s) {sorted(missing)}; expected {sorted(required)}"
        )
    table.columns = [str(column).strip().lower() for column in table.columns]

    table["population"] = pd.to_numeric(table["population"], errors="coerce")
    table["area_sq_km"] = pd.to_numeric(table["area_sq_km"], errors="coerce")
    table["year"] = pd.to_numeric(table["year"], errors="coerce").astype("Int64")

    invalid = table[table[["population", "area_sq_km", "year"]].isna().any(axis=1)]
    if not invalid.empty:
        raise SourceError(
            f"{path.name}: {len(invalid)} row(s) have non-numeric year, population or area, "
            f"e.g. {invalid.head(2).to_dict('records')}"
        )
    if (table["area_sq_km"] <= 0).any():
        raise SourceError(f"{path.name}: area_sq_km must be positive")
    return table


def _time_index(cfg: Config) -> pd.DatetimeIndex:
    """The complete period index the panel spans."""
    try:
        freq = GRANULARITY_FREQ[cfg.project.granularity]
    except KeyError:
        raise SourceError(f"unsupported granularity {cfg.project.granularity!r}") from None
    return pd.date_range(cfg.data.start_date, cfg.data.end_date, freq=freq)


def _interpolate_state(
    state: str, anchors: pd.DataFrame, index: pd.DatetimeIndex
) -> pd.DataFrame:
    """Expand one state's census anchors across the time index.

    Growth between anchors is geometric, which fits population better than a
    straight line. Outside the anchor range the nearest anchor is held flat rather
    than extrapolated — projecting a growth rate decades past its last observation
    invents precision the census does not support.
    """
    anchors = anchors.sort_values("year")
    years = anchors["year"].to_numpy(dtype=float)
    populations = anchors["population"].to_numpy(dtype=float)
    if (populations <= 0).any():
        raise SourceError(f"{state}: population must be positive to interpolate geometrically")

    target_years = index.year.to_numpy(dtype=float) + (index.dayofyear.to_numpy() - 1) / 365.25
    if len(years) == 1:
        population = np.full(len(index), populations[0])
    else:
        log_population = np.interp(target_years, years, np.log(populations))
        population = np.exp(log_population)

    area = float(anchors["area_sq_km"].iloc[0])
    return pd.concat(
        [
            pd.DataFrame(
                {"state": state, "date": index, "variable": "population", "value": population}
            ),
            pd.DataFrame(
                {
                    "state": state,
                    "date": index,
                    "variable": "population_density",
                    "value": population / area,
                }
            ),
        ],
        ignore_index=True,
    )