"""The canonical state registry — the piece every source depends on."""

from __future__ import annotations

from datetime import date

import pytest

from src.sources.registry import (
    CANONICAL_NAMES,
    UnknownStateError,
    adjacency,
    affected_by_boundary_changes,
    boundary_changes_within,
    normalise_state,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Orissa", "Odisha"),
        ("orissa", "Odisha"),
        ("  ODISHA  ", "Odisha"),
        ("Uttaranchal", "Uttarakhand"),
        ("Uttaranchal ", "Uttarakhand"),
        ("Pondicherry", "Puducherry"),
        ("NCT of Delhi", "Delhi"),
        ("Delhi (NCT)", "Delhi"),
        ("Jammu & Kashmir", "Jammu and Kashmir"),
        ("Jammu and Kashmir", "Jammu and Kashmir"),
        ("TamilNadu", "Tamil Nadu"),
        ("tamil nadu", "Tamil Nadu"),
        ("Chattisgarh", "Chhattisgarh"),
        ("Dadra & Nagar Haveli", "Dadra and Nagar Haveli and Daman and Diu"),
        ("A & N Islands", "Andaman and Nicobar Islands"),
    ],
)
def test_variants_normalise_to_one_canonical_name(raw: str, expected: str) -> None:
    """Every spelling a government table uses lands on the same canonical name."""
    assert normalise_state(raw) == expected


def test_unknown_state_raises_rather_than_dropping() -> None:
    """A silently dropped state produces a panel that looks complete but is not."""
    with pytest.raises(UnknownStateError, match="Atlantis"):
        normalise_state("Atlantis")


def test_normalisation_is_idempotent() -> None:
    """Canonical names must survive a second pass unchanged."""
    for name in CANONICAL_NAMES:
        assert normalise_state(normalise_state(name)) == name


def test_adjacency_is_symmetric_and_canonical() -> None:
    """A one-sided neighbour entry would make spatial lags directional."""
    graph = adjacency()
    assert set(graph) == set(CANONICAL_NAMES)
    for state, neighbours in graph.items():
        assert state not in neighbours, f"{state} borders itself"
        for neighbour in neighbours:
            assert neighbour in CANONICAL_NAMES
            assert state in graph[neighbour], f"{state}->{neighbour} is not reciprocated"


def test_islands_have_no_land_neighbours() -> None:
    """Empty is correct here, and informative for the spatial ablation."""
    assert adjacency()["Lakshadweep"] == frozenset()
    assert adjacency()["Andaman and Nicobar Islands"] == frozenset()


def test_boundary_changes_are_surfaced_not_silently_aliased() -> None:
    """Telangana's 2014 split breaks Andhra Pradesh's series; no alias can fix that."""
    changes = boundary_changes_within(date(2010, 1, 1), date(2023, 12, 31))
    parents = {change.parent for change in changes}
    assert "Andhra Pradesh" in parents
    assert "Jammu and Kashmir" in parents

    affected = affected_by_boundary_changes(
        ["Andhra Pradesh", "Kerala"], date(2010, 1, 1), date(2023, 12, 31)
    )
    assert "Andhra Pradesh" in affected
    assert "Kerala" not in affected


def test_window_excluding_a_change_reports_none() -> None:
    """A study window after the split has no break to report."""
    assert boundary_changes_within(date(2021, 1, 1), date(2023, 12, 31)) == ()