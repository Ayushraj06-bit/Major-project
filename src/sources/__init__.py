"""Upstream data feeds, one module per source.

Every source subclasses :class:`~src.sources.base.BaseDataSource` and returns the
same tidy long schema::

    [state, date, variable, value]

Concrete sources implement exactly two methods — :meth:`fetch_raw` and
:meth:`parse`. Caching, state-name normalisation, date coercion, window filtering
and schema validation all happen once in the base class, so a source is one file
and no other module needs editing when one is added.

State names are normalised through :mod:`src.sources.registry`, never inside a
parser. Indian publications disagree about Odisha/Orissa, Uttarakhand/Uttaranchal
and the Delhi variants, and a per-parser fix leaves the disagreement live in the
next source anyone adds.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd

from src.config import Config
from src.sources.base import (
    GRANULARITY_FREQ,
    LONG_SCHEMA,
    BaseDataSource,
    MissingRawDataError,
    SourceError,
    discover_sources,
    get_source,
    sources_for,
)
from src.sources.registry import (
    BOUNDARY_CHANGES,
    CANONICAL_NAMES,
    UnknownStateError,
    adjacency,
    boundary_changes_within,
    neighbours_of,
    normalise_state,
)

#: Key columns identifying one observation. Fusion joins on exactly these.
PANEL_KEYS: tuple[str, str] = ("state", "date")

__all__ = [
    "BOUNDARY_CHANGES",
    "CANONICAL_NAMES",
    "GRANULARITY_FREQ",
    "LONG_SCHEMA",
    "PANEL_KEYS",
    "BaseDataSource",
    "DataSource",
    "MissingRawDataError",
    "SourceError",
    "UnknownStateError",
    "adjacency",
    "boundary_changes_within",
    "discover_sources",
    "get_source",
    "neighbours_of",
    "normalise_state",
    "sources_for",
]


@runtime_checkable
class DataSource(Protocol):
    """One upstream feed, normalised to the canonical long schema.

    Note:
        ``name`` is a data member, so this protocol supports ``isinstance`` but
        not ``issubclass``.
    """

    #: Short snake_case identifier, matching a member of ``features.sources``.
    name: str

    def fetch(self, cfg: Config) -> pd.DataFrame:
        """Return this source's contribution to the panel.

        The returned frame must:

        * carry columns ``state``, ``date``, ``variable``, ``value`` in that order;
        * hold at most one row per ``(state, date, variable)``;
        * use canonical state names and period-start dates at the configured
          granularity — the base class guarantees both;
        * leave genuinely absent observations missing rather than imputing them.
          Deciding how to fill a gap belongs in preprocessing, where it is visible
          and testable, not in a loader.

        Static attributes such as population still return one row per
        ``(state, date)``, broadcast across the index, so fusion stays uniform.
        """
        ...