"""Canonical registry of Indian states and union territories.

The single place in the project that knows what a state is called. Every source
normalises through :func:`normalise_state`; no parser fixes names locally.
Fixing names per-parser is how a project ends up with "Odisha" in three sources
and "Orissa" in the fourth, joining to NaN without ever raising.

Three kinds of knowledge live here:

* **Names** — 28 states and 8 union territories under the post-2019 arrangement,
  plus an alias map for the historical and stylistic variants that appear in
  government publications.
* **Boundary changes** — states that split or merged inside a plausible study
  window. These are not naming problems and no alias map can repair them; see
  :data:`BOUNDARY_CHANGES`.
* **Adjacency** — which states share a land border, needed for the spatial lag
  features. Symmetrised on construction, so a one-sided entry cannot produce an
  asymmetric graph.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date
from functools import cache

#: 28 states, post-2019.
STATES: tuple[str, ...] = (
    "Andhra Pradesh",
    "Arunachal Pradesh",
    "Assam",
    "Bihar",
    "Chhattisgarh",
    "Goa",
    "Gujarat",
    "Haryana",
    "Himachal Pradesh",
    "Jharkhand",
    "Karnataka",
    "Kerala",
    "Madhya Pradesh",
    "Maharashtra",
    "Manipur",
    "Meghalaya",
    "Mizoram",
    "Nagaland",
    "Odisha",
    "Punjab",
    "Rajasthan",
    "Sikkim",
    "Tamil Nadu",
    "Telangana",
    "Tripura",
    "Uttar Pradesh",
    "Uttarakhand",
    "West Bengal",
)

#: 8 union territories, post-2020 (after the DNH/DD merger and the J&K reorganisation).
UNION_TERRITORIES: tuple[str, ...] = (
    "Andaman and Nicobar Islands",
    "Chandigarh",
    "Dadra and Nagar Haveli and Daman and Diu",
    "Delhi",
    "Jammu and Kashmir",
    "Ladakh",
    "Lakshadweep",
    "Puducherry",
)

CANONICAL_NAMES: tuple[str, ...] = tuple(sorted(STATES + UNION_TERRITORIES))

#: Variants that differ by more than punctuation or case, so slugging alone cannot
#: resolve them. Keys are matched after slugging, so "Orissa" also catches "orissa".
ALIASES: Mapping[str, str] = {
    # Renamed states.
    "orissa": "Odisha",
    "uttaranchal": "Uttarakhand",
    "pondicherry": "Puducherry",
    "pondichery": "Puducherry",
    "puduchery": "Puducherry",
    # Delhi appears under many administrative spellings.
    "nctofdelhi": "Delhi",
    "nctdelhi": "Delhi",
    "delhinct": "Delhi",
    "newdelhi": "Delhi",
    "nationalcapitalterritoryofdelhi": "Delhi",
    # Common misspellings in published tables.
    "chattisgarh": "Chhattisgarh",
    "chhatisgarh": "Chhattisgarh",
    "orrisa": "Odisha",
    "jammukashmir": "Jammu and Kashmir",
    "jk": "Jammu and Kashmir",
    "utteranchal": "Uttarakhand",
    # Pre-2020 union territories, merged into one.
    "dadraandnagarhaveli": "Dadra and Nagar Haveli and Daman and Diu",
    "damananddiu": "Dadra and Nagar Haveli and Daman and Diu",
    "dnh": "Dadra and Nagar Haveli and Daman and Diu",
    "dnhdd": "Dadra and Nagar Haveli and Daman and Diu",
    # Islands.
    "andamannicobarislands": "Andaman and Nicobar Islands",
    "andamannicobar": "Andaman and Nicobar Islands",
    "anislands": "Andaman and Nicobar Islands",
    # "A & N Islands" slugs to this, because & expands to "and" before stripping.
    "aandnislands": "Andaman and Nicobar Islands",
    # Spacing variants that survive slugging as one word anyway, listed for clarity.
    "tamilnad": "Tamil Nadu",
    "orissaodisha": "Odisha",
}


class UnknownStateError(KeyError):
    """Raised when a source emits a state name the registry does not recognise.

    Deliberately fatal. A silently dropped or mis-joined state is far more costly
    than a failed load: it produces a panel that looks complete but is missing a
    state, or worse, carries two half-populated rows for the same one.
    """


@dataclass(frozen=True)
class BoundaryChange:
    """A state that split or merged, and when.

    Not a naming problem. Before ``effective``, observations recorded against
    ``parent`` include the territory of every entry in ``children``, so a series
    spanning the date is measuring two different areas under one label. Any study
    window containing one of these must decide explicitly what to do — merge the
    children back into the parent for the whole period, start the series after the
    change, or drop the states involved.
    """

    effective: date
    parent: str
    children: tuple[str, ...]
    note: str


#: Reorganisations inside any plausible study window for this project.
BOUNDARY_CHANGES: tuple[BoundaryChange, ...] = (
    BoundaryChange(
        effective=date(2014, 6, 2),
        parent="Andhra Pradesh",
        children=("Andhra Pradesh", "Telangana"),
        note=(
            "Telangana separated from Andhra Pradesh. Pre-2014 Andhra Pradesh case "
            "counts and population include Telangana, so both series break here."
        ),
    ),
    BoundaryChange(
        effective=date(2019, 10, 31),
        parent="Jammu and Kashmir",
        children=("Jammu and Kashmir", "Ladakh"),
        note=(
            "Jammu and Kashmir reorganised into two union territories. Pre-2019 "
            "figures for the state include Ladakh."
        ),
    ),
    BoundaryChange(
        effective=date(2020, 1, 26),
        parent="Dadra and Nagar Haveli and Daman and Diu",
        children=("Dadra and Nagar Haveli and Daman and Diu",),
        note=(
            "Dadra and Nagar Haveli merged with Daman and Diu. Earlier sources "
            "report the two separately; the alias map folds both into the merged "
            "name, which double-counts nothing but changes the unit of observation."
        ),
    ),
)

# Neighbours by land border. Written one-directionally for legibility and
# symmetrised in _build_adjacency, so a missing reverse entry cannot silently
# create a directed edge. Island territories intentionally have none.
_NEIGHBOURS: Mapping[str, tuple[str, ...]] = {
    "Andhra Pradesh": (
        "Odisha", "Chhattisgarh", "Telangana", "Karnataka", "Tamil Nadu", "Puducherry",
    ),
    "Arunachal Pradesh": ("Assam", "Nagaland"),
    "Assam": (
        "Arunachal Pradesh", "Nagaland", "Manipur", "Mizoram", "Tripura", "Meghalaya",
        "West Bengal",
    ),
    "Bihar": ("Uttar Pradesh", "Jharkhand", "West Bengal"),
    "Chhattisgarh": (
        "Madhya Pradesh", "Maharashtra", "Telangana", "Andhra Pradesh", "Odisha", "Jharkhand",
        "Uttar Pradesh",
    ),
    "Goa": ("Maharashtra", "Karnataka"),
    "Gujarat": (
        "Rajasthan", "Madhya Pradesh", "Maharashtra", "Dadra and Nagar Haveli and Daman and Diu",
    ),
    "Haryana": (
        "Punjab", "Himachal Pradesh", "Uttarakhand", "Uttar Pradesh", "Rajasthan", "Delhi",
        "Chandigarh",
    ),
    "Himachal Pradesh": ("Jammu and Kashmir", "Ladakh", "Punjab", "Haryana", "Uttarakhand"),
    "Jharkhand": ("Bihar", "Uttar Pradesh", "Chhattisgarh", "Odisha", "West Bengal"),
    "Karnataka": ("Goa", "Maharashtra", "Telangana", "Andhra Pradesh", "Tamil Nadu", "Kerala"),
    "Kerala": ("Karnataka", "Tamil Nadu", "Puducherry"),
    "Madhya Pradesh": ("Uttar Pradesh", "Chhattisgarh", "Maharashtra", "Gujarat", "Rajasthan"),
    "Maharashtra": (
        "Gujarat", "Madhya Pradesh", "Chhattisgarh", "Telangana", "Karnataka", "Goa",
        "Dadra and Nagar Haveli and Daman and Diu",
    ),
    "Manipur": ("Nagaland", "Mizoram", "Assam"),
    "Meghalaya": ("Assam",),
    "Mizoram": ("Assam", "Manipur", "Tripura"),
    "Nagaland": ("Arunachal Pradesh", "Assam", "Manipur"),
    "Odisha": ("West Bengal", "Jharkhand", "Chhattisgarh", "Andhra Pradesh"),
    "Punjab": ("Jammu and Kashmir", "Himachal Pradesh", "Haryana", "Rajasthan", "Chandigarh"),
    "Rajasthan": ("Punjab", "Haryana", "Uttar Pradesh", "Madhya Pradesh", "Gujarat"),
    "Sikkim": ("West Bengal",),
    "Tamil Nadu": ("Andhra Pradesh", "Karnataka", "Kerala", "Puducherry"),
    "Telangana": ("Maharashtra", "Chhattisgarh", "Andhra Pradesh", "Karnataka"),
    "Tripura": ("Assam", "Mizoram"),
    "Uttar Pradesh": (
        "Uttarakhand", "Himachal Pradesh", "Haryana", "Delhi", "Rajasthan", "Madhya Pradesh",
        "Chhattisgarh", "Jharkhand", "Bihar",
    ),
    "Uttarakhand": ("Himachal Pradesh", "Uttar Pradesh"),
    "West Bengal": ("Sikkim", "Bihar", "Jharkhand", "Odisha", "Assam"),
    "Chandigarh": ("Punjab", "Haryana"),
    "Dadra and Nagar Haveli and Daman and Diu": ("Gujarat", "Maharashtra"),
    "Delhi": ("Haryana", "Uttar Pradesh"),
    "Jammu and Kashmir": ("Ladakh", "Himachal Pradesh", "Punjab"),
    "Ladakh": ("Jammu and Kashmir", "Himachal Pradesh"),
    # Non-contiguous: Puducherry and Karaikal sit in Tamil Nadu, Yanam in Andhra
    # Pradesh, Mahe in Kerala.
    "Puducherry": ("Tamil Nadu", "Andhra Pradesh", "Kerala"),
    "Andaman and Nicobar Islands": (),
    "Lakshadweep": (),
}


def slug(name: str) -> str:
    """Reduce a state name to a comparison key.

    Case, punctuation, spacing and ampersands all vary between publications;
    stripping them means only genuinely different words need an alias entry.
    """
    lowered = name.strip().lower().replace("&", " and ")
    return re.sub(r"[^a-z0-9]+", "", lowered)


@cache
def _lookup_table() -> Mapping[str, str]:
    """Slug to canonical name, covering both canonical spellings and aliases."""
    table = {slug(name): name for name in CANONICAL_NAMES}
    for alias, canonical in ALIASES.items():
        if canonical not in CANONICAL_NAMES:
            raise ValueError(f"alias {alias!r} maps to unknown canonical name {canonical!r}")
        table[slug(alias)] = canonical
    return table


def normalise_state(name: str) -> str:
    """Map any recognised spelling of a state to its canonical name.

    Args:
        name: A state name as it appears in a source file.

    Returns:
        The canonical name.

    Raises:
        UnknownStateError: the name is not recognised. Add it to :data:`ALIASES`
            rather than patching the calling parser.
    """
    if not isinstance(name, str) or not name.strip():
        raise UnknownStateError(f"expected a non-empty state name, got {name!r}")
    key = slug(name)
    try:
        return _lookup_table()[key]
    except KeyError:
        raise UnknownStateError(
            f"unrecognised state name {name!r} (normalised to {key!r}). "
            "Add it to ALIASES in src/sources/registry.py — do not correct it in the parser."
        ) from None


def is_known_state(name: str) -> bool:
    """Whether :func:`normalise_state` would succeed for this name."""
    return slug(name) in _lookup_table()


@cache
def adjacency() -> Mapping[str, frozenset[str]]:
    """Symmetric land-border graph over canonical names.

    Returns:
        Canonical name to the set of its neighbours. Island territories map to an
        empty set — they contribute no spatial lag, which is itself informative
        for the spatial ablation.
    """
    built: dict[str, set[str]] = {name: set() for name in CANONICAL_NAMES}
    for raw_state, raw_neighbours in _NEIGHBOURS.items():
        state = normalise_state(raw_state)
        for raw_neighbour in raw_neighbours:
            neighbour = normalise_state(raw_neighbour)
            if neighbour == state:
                raise ValueError(f"{state!r} is listed as its own neighbour")
            built[state].add(neighbour)
            built[neighbour].add(state)
    return {name: frozenset(neighbours) for name, neighbours in built.items()}


def neighbours_of(state: str) -> frozenset[str]:
    """Land neighbours of one state, by canonical name."""
    return adjacency()[normalise_state(state)]


def boundary_changes_within(start: date, end: date) -> tuple[BoundaryChange, ...]:
    """Reorganisations taking effect inside ``[start, end]``.

    Used by the data-quality report to surface breaks that no amount of name
    normalisation can fix.
    """
    return tuple(change for change in BOUNDARY_CHANGES if start <= change.effective <= end)


def affected_by_boundary_changes(states: Iterable[str], start: date, end: date) -> frozenset[str]:
    """States whose series is discontinuous inside the window, by canonical name."""
    requested = {normalise_state(state) for state in states}
    affected: set[str] = set()
    for change in boundary_changes_within(start, end):
        involved = {change.parent, *change.children}
        affected |= requested & involved
    return frozenset(affected)
