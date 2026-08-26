"""Fusion of several sources into one wide panel."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import src.panel as panel_module
from src.config import Config
from src.panel import PanelError, assemble_panel


class _FakeSource:
    """A source returning a long frame the test supplies directly."""

    def __init__(self, name: str, frame: pd.DataFrame) -> None:
        self.name = name
        self._frame = frame

    def fetch(self, cfg: Config) -> pd.DataFrame:
        return self._frame


def _long(states: list[str], dates: list[str], variable: str, values: list[float]) -> pd.DataFrame:
    rows = [
        {"state": state, "date": pd.Timestamp(date), "variable": variable, "value": value}
        for state, date, value in zip(states, dates, values, strict=True)
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def two_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install two fake sources covering different, partly-overlapping cells."""
    cases = _long(
        ["Kerala", "Kerala", "Odisha"],
        ["2015-01-01", "2015-02-01", "2015-01-01"],
        "cases",
        [10.0, 20.0, 30.0],
    )
    rainfall = _long(
        ["Kerala", "Odisha", "Odisha"],
        ["2015-01-01", "2015-01-01", "2015-02-01"],
        "rainfall",
        [100.0, 200.0, 300.0],
    )
    monkeypatch.setattr(
        panel_module,
        "sources_for",
        lambda cfg: (_FakeSource("cases", cases), _FakeSource("climate", rainfall)),
    )


def test_sources_fuse_on_state_and_date(cfg: Config, two_sources: None) -> None:
    """Two feeds land in one row when they describe the same state and period."""
    fused = assemble_panel(cfg)

    assert list(fused.index.names) == ["state", "date"]
    assert set(fused.columns) == {"cases", "rainfall"}
    row = fused.loc[("Kerala", pd.Timestamp("2015-01-01"))]
    assert row["cases"] == 10.0
    assert row["rainfall"] == 100.0


def test_panel_spans_the_complete_grid_not_just_reported_cells(
    cfg: Config, two_sources: None
) -> None:
    """Unreported periods must be visible NaN rows, not absent ones."""
    fused = assemble_panel(cfg)

    # 3 configured states x 36 months, regardless of what the sources returned.
    assert len(fused) == 3 * 36
    assert set(fused.index.get_level_values("state")) == {"Kerala", "Odisha", "Tamil Nadu"}
    assert fused.loc[("Tamil Nadu", slice(None))].isna().all().all()


def test_a_cell_reported_by_only_one_source_keeps_the_other_missing(
    cfg: Config, two_sources: None
) -> None:
    """Partial coverage must not be silently filled by the fusion step."""
    fused = assemble_panel(cfg)
    row = fused.loc[("Kerala", pd.Timestamp("2015-02-01"))]
    assert row["cases"] == 20.0
    assert np.isnan(row["rainfall"])


def test_two_sources_claiming_one_variable_is_rejected(
    cfg: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Otherwise one source's values would silently overwrite the other's."""
    frame = _long(["Kerala"], ["2015-01-01"], "cases", [1.0])
    monkeypatch.setattr(
        panel_module,
        "sources_for",
        lambda cfg: (_FakeSource("cases", frame), _FakeSource("other", frame)),
    )
    with pytest.raises(PanelError, match="emitted by both"):
        assemble_panel(cfg)


def test_no_rows_at_all_is_an_error_not_an_empty_panel(
    cfg: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty panel would fail much later, somewhere less informative."""
    empty = pd.DataFrame(columns=["state", "date", "variable", "value"])
    monkeypatch.setattr(panel_module, "sources_for", lambda cfg: (_FakeSource("cases", empty),))
    with pytest.raises(PanelError, match="no rows returned"):
        assemble_panel(cfg)


def test_assembled_panel_is_sorted_ready_for_lagging(cfg: Config, two_sources: None) -> None:
    """build_features rejects an unsorted panel, so fusion must deliver one sorted."""
    assert assemble_panel(cfg).index.is_monotonic_increasing