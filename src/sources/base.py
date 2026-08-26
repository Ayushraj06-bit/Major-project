"""Shared machinery for every data source.

A concrete source implements two methods — where its raw bytes come from, and how
to turn them into long rows. Everything else happens once, here: caching, state
normalisation, date filtering, schema validation, and registration.

Adding a source is therefore one new file in this package. It is discovered
automatically, so no registry list, import block or dispatch table needs editing
elsewhere.
"""

from __future__ import annotations

import importlib
import pkgutil
from abc import ABC, abstractmethod
from collections.abc import Mapping
from datetime import date
from pathlib import Path
from typing import Any, ClassVar

import pandas as pd

from src.config import Config, load_config
from src.io import cached
from src.sources.registry import normalise_state

#: Column order every source returns, before fusion pivots it wide.
LONG_SCHEMA: tuple[str, str, str, str] = ("state", "date", "variable", "value")

#: pandas offset alias for each supported granularity, anchored to period start.
GRANULARITY_FREQ: Mapping[str, str] = {"monthly": "MS", "weekly": "W-MON"}

_REGISTRY: dict[str, type[BaseDataSource]] = {}


class SourceError(RuntimeError):
    """Raised when a source cannot produce a valid slice of the panel."""


class MissingRawDataError(SourceError, FileNotFoundError):
    """Raised when a required download or manual export is absent from ``data/raw/``.

    Carries the exact path and provenance, because acquiring these files is manual
    for most Indian dengue sources and the error message is the instruction.
    """


class BaseDataSource(ABC):
    """Template for one upstream feed.

    Subclasses set :attr:`name` and :attr:`variables` and implement
    :meth:`fetch_raw` and :meth:`parse`. They must not normalise state names,
    filter dates, or touch the cache — those are handled here so that every source
    behaves identically and a bug is fixed in one place.
    """

    #: Snake_case identifier. Must match a member of ``features.sources``.
    name: ClassVar[str]
    #: Variable names this source emits into the ``variable`` column.
    variables: ClassVar[tuple[str, ...]]
    #: Human-readable provenance, quoted in error messages and the report.
    provenance: ClassVar[str] = ""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Register concrete subclasses by name as they are defined."""
        super().__init_subclass__(**kwargs)
        if getattr(cls, "__abstractmethods__", None):
            return
        for attribute in ("name", "variables"):
            if not getattr(cls, attribute, None):
                raise TypeError(f"{cls.__name__} must set a non-empty '{attribute}'")
        existing = _REGISTRY.get(cls.name)
        if existing is not None and existing is not cls:
            raise TypeError(
                f"source name {cls.name!r} is claimed by both "
                f"{existing.__name__} and {cls.__name__}"
            )
        _REGISTRY[cls.name] = cls

    # -- the two hooks a concrete source implements -------------------------- #

    @abstractmethod
    def fetch_raw(self, cfg: Config) -> Any:
        """Obtain the immutable raw input, downloading it into ``data/raw/`` if needed.

        Whatever is returned is handed straight to :meth:`parse` — a path, a list
        of paths, or an in-memory object. Files under ``data/raw/`` are never
        modified in place; derived data belongs in ``data/interim/``.

        Raises:
            MissingRawDataError: the input must be acquired manually and is absent.
        """

    @abstractmethod
    def parse(self, raw: Any, cfg: Config) -> pd.DataFrame:
        """Convert the raw input to long rows.

        Must return columns ``state``, ``date``, ``variable``, ``value``. State
        names may be in whatever spelling the source uses — the base class
        normalises them. Genuinely absent observations are left out or set NaN,
        never imputed; filling gaps is a feature-engineering decision that belongs
        somewhere visible and testable.
        """

    # -- everything below is shared and should not be overridden ------------- #

    def fetch(self, cfg: Config) -> pd.DataFrame:
        """Return this source's validated, cached contribution to the panel."""
        return _fetch_source(
            source_name=self.name,
            states=tuple(cfg.data.states),
            start_date=cfg.data.start_date,
            end_date=cfg.data.end_date,
            granularity=cfg.project.granularity,
        )

    def raw_dir(self, cfg: Config) -> Path:
        """This source's immutable download directory under ``data/raw/``."""
        return Path(cfg.paths.data_raw) / self.name

    def require_raw_file(self, cfg: Config, filename: str, how: str) -> Path:
        """Return a required raw file, or explain precisely how to obtain it.

        Args:
            cfg: Loaded configuration.
            filename: Expected filename inside :meth:`raw_dir`.
            how: Acquisition instructions, shown when the file is absent.

        Raises:
            MissingRawDataError: the file is not there.
        """
        path = self.raw_dir(cfg) / filename
        if not path.is_file():
            raise MissingRawDataError(
                f"[{self.name}] required raw file is missing:\n"
                f"  expected at: {path}\n"
                f"  provenance : {self.provenance or 'see module docstring'}\n"
                f"  how to get : {how}"
            )
        return path


# --------------------------------------------------------------------------- #
# Registry and discovery
# --------------------------------------------------------------------------- #


def discover_sources() -> Mapping[str, type[BaseDataSource]]:
    """Import every module in this package so subclasses self-register.

    This is what makes a new source a one-file change: drop it in
    ``src/sources/``, and it appears here without any list being edited.
    """
    package = importlib.import_module(__package__)
    for module in pkgutil.iter_modules(package.__path__):
        if not module.name.startswith("_"):
            importlib.import_module(f"{__package__}.{module.name}")
    return dict(_REGISTRY)


def get_source(name: str) -> BaseDataSource:
    """Instantiate the registered source with this name.

    Raises:
        KeyError: no source claims the name.
    """
    registry = discover_sources()
    try:
        return registry[name]()
    except KeyError:
        raise KeyError(
            f"unknown source {name!r}; registered sources are {sorted(registry)}"
        ) from None


def sources_for(cfg: Config) -> tuple[BaseDataSource, ...]:
    """Instantiate the sources listed in ``data.sources``.

    Deliberately **not** ``features.sources``. The ablation selects which sources
    feed the model, not which are loaded: the target is built from cases and
    population, so the climate-only configuration still needs both on the panel.
    Ablating at fetch time would leave configuration A with nothing to predict.
    """
    return tuple(get_source(name) for name in cfg.data.sources)


# --------------------------------------------------------------------------- #
# The cached pipeline every source runs through
# --------------------------------------------------------------------------- #


@cached("source", key_args=("source_name",))
def _fetch_source(
    source_name: str,
    states: tuple[str, ...],
    start_date: date,
    end_date: date,
    granularity: str,
) -> pd.DataFrame:
    """Fetch, parse, normalise, filter and validate one source.

    The parameters are the cache key: they are exactly the configuration that can
    change the result. The full config is reloaded inside rather than passed, as
    it is not fingerprintable.
    """
    cfg = load_config()
    source = get_source(source_name)

    parsed = source.parse(source.fetch_raw(cfg), cfg)
    frame = _validate_columns(parsed, source)
    frame = _normalise_states(frame, source, keep=states)
    frame = _coerce_dates(frame, source, granularity=granularity)
    frame = _filter_window(frame, start_date=start_date, end_date=end_date)
    _validate_content(frame, source)
    return frame.sort_values(list(LONG_SCHEMA)).reset_index(drop=True)


def _validate_columns(frame: pd.DataFrame, source: BaseDataSource) -> pd.DataFrame:
    """Check the parser returned the long schema, and drop anything extra."""
    if not isinstance(frame, pd.DataFrame):
        raise SourceError(
            f"[{source.name}] parse() must return a DataFrame, got {type(frame).__name__}"
        )
    missing = [column for column in LONG_SCHEMA if column not in frame.columns]
    if missing:
        raise SourceError(
            f"[{source.name}] parse() output is missing column(s) {missing}; "
            f"expected {list(LONG_SCHEMA)}, got {list(frame.columns)}"
        )
    return frame.loc[:, list(LONG_SCHEMA)].copy()


def _normalise_states(
    frame: pd.DataFrame, source: BaseDataSource, keep: tuple[str, ...]
) -> pd.DataFrame:
    """Canonicalise state names, then restrict to the states under study.

    Normalisation happens before filtering so that a source spelling a wanted
    state differently is still kept, rather than silently vanishing.
    """
    try:
        frame["state"] = frame["state"].map(normalise_state)
    except Exception as exc:
        raise SourceError(f"[{source.name}] state normalisation failed: {exc}") from exc
    wanted = {normalise_state(state) for state in keep}
    return frame.loc[frame["state"].isin(wanted)]


def _coerce_dates(frame: pd.DataFrame, source: BaseDataSource, granularity: str) -> pd.DataFrame:
    """Parse dates and snap them to the start of their period.

    Sources label a month variously as the 1st, the last day, or a bare
    "2019-07". Snapping to period start makes them join.
    """
    parsed = pd.to_datetime(frame["date"], errors="coerce")
    unparseable = int(parsed.isna().sum() - frame["date"].isna().sum())
    if unparseable > 0:
        examples = frame.loc[parsed.isna() & frame["date"].notna(), "date"].head(3).tolist()
        raise SourceError(
            f"[{source.name}] {unparseable} date value(s) could not be parsed, e.g. {examples}"
        )
    frame = frame.copy()
    frame["date"] = parsed.dt.to_period(_period_alias(granularity)).dt.start_time
    return frame


def _filter_window(frame: pd.DataFrame, start_date: date, end_date: date) -> pd.DataFrame:
    """Restrict to the configured study window."""
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    return frame.loc[frame["date"].between(start, end)]


def _validate_content(frame: pd.DataFrame, source: BaseDataSource) -> None:
    """Reject output that would corrupt the panel downstream."""
    declared = set(source.variables)
    emitted = set(frame["variable"].unique())
    undeclared = sorted(emitted - declared)
    if undeclared:
        raise SourceError(
            f"[{source.name}] emitted undeclared variable(s) {undeclared}; "
            f"declared variables are {sorted(declared)}"
        )

    duplicated = frame.duplicated(subset=["state", "date", "variable"], keep=False)
    if duplicated.any():
        examples = (
            frame.loc[duplicated, ["state", "date", "variable"]].head(3).to_dict("records")
        )
        raise SourceError(
            f"[{source.name}] {int(duplicated.sum())} duplicate (state, date, variable) row(s), "
            f"e.g. {examples}. The panel cannot pivot with duplicates; aggregate in parse()."
        )

    if not pd.api.types.is_numeric_dtype(frame["value"]):
        raise SourceError(
            f"[{source.name}] 'value' must be numeric, got dtype {frame['value'].dtype}"
        )


def _period_alias(granularity: str) -> str:
    """Pandas period alias for a configured granularity."""
    try:
        return {"monthly": "M", "weekly": "W-MON"}[granularity]
    except KeyError:
        raise SourceError(f"unsupported granularity {granularity!r}") from None
