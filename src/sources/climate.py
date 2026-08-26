"""Climate drivers aggregated from gridded reanalysis to state level.

ERA5 is gridded, so turning it into a state series needs a spatial aggregation,
and the choice of aggregation matters. A flat mean over a state's cells is
dominated by whatever area is largest, which in Rajasthan or Ladakh is mostly
empty. Dengue transmission happens where people are, so cells are weighted by
population:

    state_value(t) = sum_c w_c * value_c(t),    w_c = pop_c / sum_c pop_c

The weighting itself is the part worth testing, so :func:`aggregate_to_states` is
a pure function over arrays, independent of ERA5, NetCDF or the download.

Downloading needs a CDS account and ``~/.cdsapirc``. When ``cdsapi`` is absent or
unconfigured the source falls back to whatever NetCDF files are already in
``data/raw/climate/``, and explains what to do if there are none.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import Config
from src.sources.base import BaseDataSource, SourceError

#: ERA5 short names mapped to the variable names used in the panel.
ERA5_VARIABLES: dict[str, str] = {
    "total_precipitation": "rainfall",
    "2m_temperature": "temperature",
    "2m_dewpoint_temperature": "dewpoint",
}

_ACQUISITION = (
    "Either install and configure the CDS API (pip install cdsapi, then create "
    "~/.cdsapirc with your key from https://cds.climate.copernicus.eu/), or "
    "download ERA5 monthly means for India manually and place the .nc file(s) in "
    "the directory above."
)


class ClimateSource(BaseDataSource):
    """Rainfall, temperature and humidity per state, population-weighted."""

    name = "climate"
    variables = ("rainfall", "temperature", "humidity")
    provenance = "ERA5 reanalysis (Copernicus CDS), population-weighted to state level"

    def fetch_raw(self, cfg: Config) -> list[Path]:
        """Return the NetCDF files to aggregate, downloading them if possible."""
        directory = self.raw_dir(cfg)
        existing = sorted(directory.glob("*.nc"))
        if existing:
            return existing

        directory.mkdir(parents=True, exist_ok=True)
        downloaded = _download_era5(directory, cfg)
        if downloaded:
            return downloaded
        return [self.require_raw_file(cfg, "era5_india.nc", _ACQUISITION)]

    def parse(self, raw: Sequence[Path], cfg: Config) -> pd.DataFrame:
        """Aggregate gridded fields to state series."""
        grids = _open_grids(raw)
        weights = _population_weights(cfg, grids)
        frames = [
            aggregate_to_states(
                values=grid.values,
                dates=grid.dates,
                weights=weights,
                variable=grid.variable,
            )
            for grid in grids
        ]
        combined = pd.concat(frames, ignore_index=True)
        return _derive_humidity(combined)


# --------------------------------------------------------------------------- #
# The aggregation — pure, and the part that is actually tested
# --------------------------------------------------------------------------- #


def aggregate_to_states(
    values: np.ndarray,
    dates: Sequence[pd.Timestamp],
    weights: pd.DataFrame,
    variable: str,
) -> pd.DataFrame:
    """Collapse a gridded time series to one weighted series per state.

    Args:
        values: ``(n_times, n_cells)`` field values, cells in a fixed order.
        dates: ``n_times`` timestamps.
        weights: Long frame with columns ``state``, ``cell``, ``weight``. Weights
            must sum to 1 within each state; ``cell`` indexes the second axis of
            ``values``.
        variable: Name to emit in the ``variable`` column.

    Returns:
        Long rows of ``state``, ``date``, ``variable``, ``value``.

    Raises:
        SourceError: shapes disagree, a cell index is out of range, or some
            state's weights do not sum to 1.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise SourceError(
            f"expected a (times, cells) array for {variable!r}, got shape {values.shape}"
        )
    if values.shape[0] != len(dates):
        raise SourceError(
            f"{variable!r}: {values.shape[0]} time steps but {len(dates)} dates supplied"
        )

    required = {"state", "cell", "weight"}
    missing = required - set(weights.columns)
    if missing:
        raise SourceError(f"weights frame is missing column(s) {sorted(missing)}")

    n_cells = values.shape[1]
    out_of_range = weights.loc[(weights["cell"] < 0) | (weights["cell"] >= n_cells), "cell"]
    if not out_of_range.empty:
        raise SourceError(
            f"{variable!r}: weight cell index {out_of_range.iloc[0]} "
            f"outside grid of {n_cells} cells"
        )

    totals = weights.groupby("state")["weight"].sum()
    unnormalised = totals[(totals - 1.0).abs() > 1e-6]
    if not unnormalised.empty:
        raise SourceError(
            f"weights must sum to 1 per state; {unnormalised.index[0]!r} sums to "
            f"{unnormalised.iloc[0]:.6f}"
        )

    records: list[pd.DataFrame] = []
    for state, group in weights.groupby("state", sort=True):
        # (times, cells_of_state) @ (cells_of_state,) -> (times,)
        series = values[:, group["cell"].to_numpy()] @ group["weight"].to_numpy()
        records.append(
            pd.DataFrame(
                {"state": state, "date": list(dates), "variable": variable, "value": series}
            )
        )
    return pd.concat(records, ignore_index=True)


def population_weights_from_grid(
    cell_states: Sequence[str], cell_population: Sequence[float]
) -> pd.DataFrame:
    """Build normalised per-state cell weights from a cell-to-state assignment.

    Args:
        cell_states: State owning each grid cell, in cell order. Empty string or
            ``None`` for cells outside every state of interest.
        cell_population: Population in each cell, same order.

    Returns:
        Long frame of ``state``, ``cell``, ``weight``, weights summing to 1 per state.

    Raises:
        SourceError: lengths differ, a population is negative, or a state's cells
            hold no population at all.
    """
    if len(cell_states) != len(cell_population):
        raise SourceError(
            f"cell_states ({len(cell_states)}) and cell_population "
            f"({len(cell_population)}) must be the same length"
        )
    frame = pd.DataFrame(
        {
            "cell": np.arange(len(cell_states)),
            "state": [state if state else None for state in cell_states],
            "population": np.asarray(cell_population, dtype=float),
        }
    ).dropna(subset=["state"])

    if (frame["population"] < 0).any():
        raise SourceError("cell populations must be non-negative")

    totals = frame.groupby("state")["population"].transform("sum")
    empty = frame.loc[totals == 0, "state"].unique()
    if len(empty):
        raise SourceError(
            f"state(s) {sorted(empty)} have zero population across all their cells, "
            "so a population-weighted mean is undefined for them"
        )
    frame["weight"] = frame["population"] / totals
    return frame.loc[:, ["state", "cell", "weight"]].reset_index(drop=True)


def relative_humidity(temperature_c: np.ndarray, dewpoint_c: np.ndarray) -> np.ndarray:
    """Relative humidity in percent, via the Magnus approximation.

    ERA5 supplies dewpoint rather than relative humidity, so it is derived here
    instead of being requested as a separate field.
    """
    magnus_a, magnus_b = 17.625, 243.04
    numerator = np.exp(magnus_a * dewpoint_c / (magnus_b + dewpoint_c))
    denominator = np.exp(magnus_a * temperature_c / (magnus_b + temperature_c))
    return 100.0 * numerator / denominator


# --------------------------------------------------------------------------- #
# NetCDF and CDS plumbing — optional dependencies, imported where used
# --------------------------------------------------------------------------- #


class _Grid:
    """One gridded variable read from NetCDF."""

    def __init__(self, variable: str, values: np.ndarray, dates: list[pd.Timestamp]) -> None:
        self.variable = variable
        self.values = values
        self.dates = dates


def _open_grids(paths: Sequence[Path]) -> list[_Grid]:
    """Read ERA5 NetCDF files into flattened ``(times, cells)`` arrays."""
    try:
        import xarray as xr
    except ImportError as exc:
        raise SourceError(
            "reading ERA5 NetCDF needs xarray and netCDF4: pip install xarray netCDF4"
        ) from exc

    grids: list[_Grid] = []
    dataset = xr.open_mfdataset([str(path) for path in paths], combine="by_coords")
    try:
        dates = [pd.Timestamp(value) for value in dataset["time"].values]
        for short_name, variable in ERA5_VARIABLES.items():
            if short_name not in dataset:
                continue
            array = dataset[short_name]
            flattened = array.values.reshape(array.shape[0], -1)
            grids.append(_Grid(variable, _to_panel_units(variable, flattened), dates))
    finally:
        dataset.close()

    if not grids:
        raise SourceError(
            f"none of {sorted(ERA5_VARIABLES)} found in {[p.name for p in paths]}"
        )
    return grids


def _to_panel_units(variable: str, values: np.ndarray) -> np.ndarray:
    """Convert ERA5 SI units to the units the panel reports."""
    if variable in {"temperature", "dewpoint"}:
        return values - 273.15  # kelvin to celsius
    if variable == "rainfall":
        return values * 1000.0  # metres to millimetres
    return values


def _derive_humidity(frame: pd.DataFrame) -> pd.DataFrame:
    """Replace the dewpoint rows with relative humidity, keeping the rest."""
    if "dewpoint" not in set(frame["variable"]):
        return frame

    wide = frame.pivot_table(index=["state", "date"], columns="variable", values="value")
    if "temperature" not in wide.columns:
        raise SourceError("dewpoint present but temperature missing; cannot derive humidity")

    humidity = pd.DataFrame(
        {
            "value": relative_humidity(
                wide["temperature"].to_numpy(), wide["dewpoint"].to_numpy()
            )
        },
        index=wide.index,
    ).reset_index()
    humidity["variable"] = "humidity"

    kept = frame.loc[frame["variable"] != "dewpoint"]
    return pd.concat([kept, humidity.loc[:, list(kept.columns)]], ignore_index=True)


def _population_weights(cfg: Config, grids: Sequence[_Grid]) -> pd.DataFrame:
    """Load the cell-to-state assignment produced during raw data preparation.

    Building this mapping requires state boundary polygons and a gridded
    population raster, which is a one-off preparation step rather than part of
    every run. The result is cached as a CSV beside the NetCDF files.
    """
    directory = Path(cfg.paths.data_raw) / "climate"
    path = directory / "cell_state_population.csv"
    if not path.is_file():
        raise SourceError(
            f"[climate] population weighting needs a cell-to-state mapping at {path}.\n"
            "  Expected columns: cell,state,population — one row per ERA5 grid cell,\n"
            "  produced once by overlaying state boundaries (e.g. GADM or Survey of\n"
            "  India) and a gridded population raster (e.g. WorldPop or GPW) on the\n"
            "  ERA5 grid. A flat unweighted mean is not an acceptable substitute:\n"
            "  it weights empty land as heavily as cities."
        )
    table = pd.read_csv(path)
    return population_weights_from_grid(
        cell_states=table["state"].fillna("").tolist(),
        cell_population=table["population"].tolist(),
    )


def _download_era5(directory: Path, cfg: Config) -> list[Path]:
    """Request ERA5 monthly means from the CDS, or return nothing if unavailable."""
    try:
        import cdsapi
    except ImportError:
        return []

    target = directory / "era5_india.nc"
    try:
        client = cdsapi.Client()
        client.retrieve(
            "reanalysis-era5-single-levels-monthly-means",
            {
                "product_type": "monthly_averaged_reanalysis",
                "variable": list(ERA5_VARIABLES),
                "year": [
                    str(year)
                    for year in range(cfg.data.start_date.year, cfg.data.end_date.year + 1)
                ],
                "month": [f"{month:02d}" for month in range(1, 13)],
                "time": "00:00",
                # India bounding box: north, west, south, east.
                "area": [37.5, 68.0, 6.5, 97.5],
                "format": "netcdf",
            },
            str(target),
        )
    except Exception as exc:  # noqa: BLE001 - any CDS failure falls back to manual acquisition
        raise SourceError(
            f"[climate] CDS download failed: {exc}\n{_ACQUISITION}"
        ) from exc
    return [target] if target.is_file() else []