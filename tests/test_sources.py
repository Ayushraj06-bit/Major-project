"""Base-class guarantees, source discovery, and the climate aggregation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.config import Config
from src.sources.base import (
    LONG_SCHEMA,
    BaseDataSource,
    SourceError,
    _coerce_dates,
    _normalise_states,
    _validate_columns,
    _validate_content,
    discover_sources,
)
from src.sources.climate import (
    aggregate_to_states,
    population_weights_from_grid,
    relative_humidity,
)


class _Fake(BaseDataSource):
    """A source whose parse output the tests control directly."""

    name = "fake_for_tests"
    variables = ("cases",)

    def __init__(self, frame: pd.DataFrame | None = None) -> None:
        self._frame = frame

    def fetch_raw(self, cfg: Config) -> Any:
        return None

    def parse(self, raw: Any, cfg: Config) -> pd.DataFrame:
        assert self._frame is not None
        return self._frame


def _long(**overrides: Any) -> pd.DataFrame:
    base = {
        "state": ["Orissa", "Kerala"],
        "date": ["2015-01-15", "2015-02-28"],
        "variable": ["cases", "cases"],
        "value": [10.0, 20.0],
    }
    base.update(overrides)
    return pd.DataFrame(base)


# --------------------------------------------------------------------------- #
# Registration and discovery
# --------------------------------------------------------------------------- #


def test_all_four_sources_are_discovered_without_a_registry_list() -> None:
    """Adding a source is one file; nothing else should need editing."""
    discovered = discover_sources()
    assert {"cases", "climate", "awareness", "demographic"} <= set(discovered)


def test_duplicate_source_name_is_rejected_at_definition() -> None:
    """Two sources claiming one name would make get_source non-deterministic."""
    with pytest.raises(TypeError, match="claimed by both"):

        class _Clash(BaseDataSource):
            name = "cases"
            variables = ("cases",)

            def fetch_raw(self, cfg: Config) -> Any:
                return None

            def parse(self, raw: Any, cfg: Config) -> pd.DataFrame:
                return _long()


def test_source_without_variables_is_rejected() -> None:
    """A source that declares nothing cannot be validated against."""
    with pytest.raises(TypeError, match="non-empty 'variables'"):

        class _Empty(BaseDataSource):
            name = "empty_for_tests"
            variables = ()

            def fetch_raw(self, cfg: Config) -> Any:
                return None

            def parse(self, raw: Any, cfg: Config) -> pd.DataFrame:
                return _long()


# --------------------------------------------------------------------------- #
# Base-class pipeline steps
# --------------------------------------------------------------------------- #


def test_parser_may_use_any_spelling_because_the_base_class_normalises() -> None:
    """Sources must not fix names themselves — this is why they need not."""
    frame = _validate_columns(_long(), _Fake())
    out = _normalise_states(frame, _Fake(), keep=("Odisha", "Kerala"))
    assert set(out["state"]) == {"Odisha", "Kerala"}


def test_states_outside_the_study_are_dropped_after_normalisation() -> None:
    """Filtering before normalising would drop 'Orissa' while wanting 'Odisha'."""
    frame = _validate_columns(_long(), _Fake())
    out = _normalise_states(frame, _Fake(), keep=("Odisha",))
    assert set(out["state"]) == {"Odisha"}


def test_missing_schema_column_is_an_error() -> None:
    with pytest.raises(SourceError, match="missing column"):
        _validate_columns(_long().drop(columns=["variable"]), _Fake())


def test_dates_snap_to_period_start_so_sources_join() -> None:
    """One source labels a month by its 15th, another by its last day."""
    frame = _validate_columns(_long(), _Fake())
    out = _coerce_dates(frame, _Fake(), granularity="monthly")
    assert list(out["date"]) == [pd.Timestamp("2015-01-01"), pd.Timestamp("2015-02-01")]


@pytest.mark.filterwarnings("ignore:Could not infer format:UserWarning")
def test_unparseable_date_is_an_error_not_a_dropped_row() -> None:
    frame = _validate_columns(_long(date=["not-a-date", "2015-02-01"]), _Fake())
    with pytest.raises(SourceError, match="could not be parsed"):
        _coerce_dates(frame, _Fake(), granularity="monthly")


def test_duplicate_observations_are_rejected_before_they_reach_the_pivot() -> None:
    """A pivot would silently keep one of them; the panel would be quietly wrong."""
    frame = _long(
        state=["Kerala", "Kerala"], date=["2015-01-01", "2015-01-01"], value=[1.0, 2.0]
    )
    with pytest.raises(SourceError, match="duplicate"):
        _validate_content(frame, _Fake())


def test_undeclared_variable_is_rejected() -> None:
    frame = _long(variable=["cases", "rainfall"])
    with pytest.raises(SourceError, match="undeclared variable"):
        _validate_content(frame, _Fake())


def test_non_numeric_values_are_rejected() -> None:
    frame = _long(value=["ten", "twenty"])
    with pytest.raises(SourceError, match="must be numeric"):
        _validate_content(frame, _Fake())


# --------------------------------------------------------------------------- #
# Climate: the population-weighted aggregation
# --------------------------------------------------------------------------- #


def test_population_weighting_follows_people_not_area() -> None:
    """The whole point: one populous cell must dominate three empty ones."""
    weights = population_weights_from_grid(
        cell_states=["Kerala", "Kerala", "Kerala", "Kerala"],
        cell_population=[900.0, 0.0, 0.0, 100.0],
    )
    # Hot city cell at 40, cold empty cells at 0.
    values = np.array([[40.0, 0.0, 0.0, 20.0]])
    out = aggregate_to_states(values, [pd.Timestamp("2015-01-01")], weights, "temperature")

    assert out["value"].iloc[0] == pytest.approx(0.9 * 40.0 + 0.1 * 20.0)
    # A flat spatial mean would give 15.0, dominated by empty land.
    assert out["value"].iloc[0] != pytest.approx(values.mean())


def test_weights_are_normalised_within_each_state() -> None:
    weights = population_weights_from_grid(
        cell_states=["Kerala", "Kerala", "Odisha"], cell_population=[3.0, 1.0, 5.0]
    )
    totals = weights.groupby("state")["weight"].sum()
    assert totals.round(9).eq(1.0).all()


def test_cells_outside_every_state_are_excluded() -> None:
    """Ocean and foreign cells carry no population weight."""
    weights = population_weights_from_grid(
        cell_states=["Kerala", "", "Kerala"], cell_population=[1.0, 999.0, 1.0]
    )
    assert set(weights["cell"]) == {0, 2}


def test_state_with_no_population_is_an_error_not_a_divide_by_zero() -> None:
    with pytest.raises(SourceError, match="zero population"):
        population_weights_from_grid(cell_states=["Ladakh"], cell_population=[0.0])


def test_aggregation_rejects_a_cell_index_outside_the_grid() -> None:
    weights = pd.DataFrame({"state": ["Kerala"], "cell": [7], "weight": [1.0]})
    with pytest.raises(SourceError, match="outside grid"):
        aggregate_to_states(np.zeros((1, 3)), [pd.Timestamp("2015-01-01")], weights, "rainfall")


def test_aggregation_rejects_unnormalised_weights() -> None:
    weights = pd.DataFrame({"state": ["Kerala", "Kerala"], "cell": [0, 1], "weight": [0.3, 0.3]})
    with pytest.raises(SourceError, match="must sum to 1"):
        aggregate_to_states(np.zeros((1, 2)), [pd.Timestamp("2015-01-01")], weights, "rainfall")


def test_aggregation_rejects_a_time_axis_mismatch() -> None:
    weights = pd.DataFrame({"state": ["Kerala"], "cell": [0], "weight": [1.0]})
    with pytest.raises(SourceError, match="time steps"):
        aggregate_to_states(np.zeros((3, 1)), [pd.Timestamp("2015-01-01")], weights, "rainfall")


def test_relative_humidity_is_100_percent_at_the_dewpoint() -> None:
    """Saturation is the one value the Magnus formula must get exactly right."""
    temperature = np.array([20.0, 30.0])
    assert relative_humidity(temperature, temperature) == pytest.approx([100.0, 100.0])


def test_relative_humidity_falls_as_air_warms_above_its_dewpoint() -> None:
    humidity = relative_humidity(np.array([30.0]), np.array([20.0]))
    assert 0.0 < humidity[0] < 100.0


def test_long_schema_is_the_documented_order() -> None:
    assert LONG_SCHEMA == ("state", "date", "variable", "value")