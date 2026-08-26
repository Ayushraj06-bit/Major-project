"""Public-awareness signal: search and encyclopaedia interest in dengue.

This is search interest, not social-media scraping. Three backends are supported,
in descending order of usefulness:

1. **Manual Google Trends CSV export** — the only reliable route to *state-level*
   interest. ``pytrends`` was archived in April 2025 and fails on first call, so
   it is not used.
2. **trendspy** — a maintained Trends client, used when installed.
3. **Wikipedia Pageviews API** — official, stable, and free, but **national only**.
   The API has no Indian-state granularity, so this backend broadcasts one national
   series to every state. That makes it a shared covariate: it can carry national
   attention dynamics but cannot distinguish Kerala from Rajasthan, and it will
   contribute nothing to a pooled model's *between-state* variation. Say so in the
   report rather than letting a reviewer find it.

A caveat applies to all three. Search interest may *lag* outbreaks rather than
lead them, because people search after news coverage. If attribution shows it
contributing only at lag 0 or negative lags, it is a nowcasting signal, not a
predictive one — which is a legitimate finding, not a failure.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import TypeVar

import pandas as pd

from src.config import Config
from src.sources.base import BaseDataSource, SourceError
from src.sources.registry import normalise_state

#: Search terms whose interest is summed into one awareness signal.
SEARCH_TERMS: tuple[str, ...] = ("dengue", "dengue symptoms", "dengue fever")

#: Wikipedia articles used by the national fallback.
WIKIPEDIA_ARTICLES: tuple[str, ...] = ("Dengue_fever", "Dengue")

_PAGEVIEWS_URL = (
    "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
    "en.wikipedia/all-access/all-agents/{article}/monthly/{start}/{end}"
)
_USER_AGENT = "dengue-forecast/0.1 (college research project)"

_ACQUISITION = (
    "Open https://trends.google.com/trends/explore?geo=IN, search 'dengue', set the "
    "date range to the study window, and download the 'Interest by subregion' CSV "
    "for each period — or install trendspy (pip install trendspy). Failing both, "
    "the Wikipedia Pageviews fallback runs automatically but is national-only."
)


class TrendsSource(BaseDataSource):
    """Public search interest in dengue, per state where the backend allows it."""

    name = "awareness"
    variables = ("search_interest",)
    provenance = "Google Trends (manual export or trendspy); Wikipedia Pageviews fallback"

    def fetch_raw(self, cfg: Config) -> tuple[str, object]:
        """Pick the best available backend and return its raw payload.

        Returns:
            A ``(backend, payload)`` pair, so :meth:`parse` knows what it is
            holding and the report can state which backend produced the series.
        """
        exports = sorted(self.raw_dir(cfg).glob("*.csv"))
        if exports:
            return ("trends_csv", exports)

        frame = _fetch_via_trendspy(cfg)
        if frame is not None:
            return ("trendspy", frame)

        return ("wikipedia", _fetch_pageviews(cfg))

    def parse(self, raw: tuple[str, object], cfg: Config) -> pd.DataFrame:
        """Normalise whichever backend produced the data into long rows."""
        backend, payload = raw
        if backend == "trends_csv":
            long = _parse_trends_exports(_expect(payload, list, backend))
        elif backend == "trendspy":
            long = _expect(payload, pd.DataFrame, backend).copy()
        elif backend == "wikipedia":
            long = _broadcast_national(_expect(payload, pd.DataFrame, backend), cfg)
        else:
            raise SourceError(f"unknown awareness backend {backend!r}")

        long["variable"] = self.variables[0]
        long["value"] = pd.to_numeric(long["value"], errors="coerce")
        return long.loc[:, ["state", "date", "variable", "value"]]


_Payload = TypeVar("_Payload")


def _expect(payload: object, kind: type[_Payload], backend: str) -> _Payload:
    """Narrow a backend payload to the type that backend promises to return.

    The backend tag and its payload are set in one place, so a mismatch means a
    coding error rather than bad input — but it should still fail here, naming the
    backend, instead of surfacing as an AttributeError three frames away.
    """
    if not isinstance(payload, kind):
        raise SourceError(
            f"[awareness] backend {backend!r} produced {type(payload).__name__}, "
            f"expected {kind.__name__}"
        )
    return payload


def _parse_trends_exports(paths: list[Path]) -> pd.DataFrame:
    """Read Google Trends 'Interest by subregion' CSVs into long rows.

    Trends exports carry two preamble lines before the header, and each file
    covers one period. The period is taken from the filename stem, which the user
    controls, so an unparseable name is an error rather than a guess.
    """
    frames: list[pd.DataFrame] = []
    for path in paths:
        period = pd.to_datetime(path.stem, errors="coerce")
        if period is pd.NaT:
            raise SourceError(
                f"{path.name}: filename must be the period it covers, e.g. '2019-07.csv'"
            )
        table = pd.read_csv(path, skiprows=2, names=["state", "value"], header=0)
        table["date"] = period
        frames.append(table)

    if not frames:
        raise SourceError("no Google Trends CSV exports found")
    return pd.concat(frames, ignore_index=True)


def _fetch_via_trendspy(cfg: Config) -> pd.DataFrame | None:
    """Query Trends through trendspy, or return None if it is unavailable."""
    try:
        from trendspy import Trends
    except ImportError:
        return None

    try:
        client = Trends()
        timeframe = f"{cfg.data.start_date.isoformat()} {cfg.data.end_date.isoformat()}"
        raw = client.interest_by_region(
            list(SEARCH_TERMS), timeframe=timeframe, geo="IN", resolution="REGION"
        )
    except Exception as exc:  # noqa: BLE001 - any client failure falls through to Wikipedia
        raise SourceError(f"[awareness] trendspy query failed: {exc}\n{_ACQUISITION}") from exc

    long = raw.reset_index().melt(id_vars=raw.index.name or "index", var_name="term",
                                  value_name="value")
    long = long.rename(columns={long.columns[0]: "state"})
    # Trends returns one column per term; sum them into a single interest signal.
    return long.groupby(["state"], as_index=False)["value"].sum().assign(date=cfg.data.start_date)


def _fetch_pageviews(cfg: Config) -> pd.DataFrame:
    """Fetch monthly Wikipedia pageviews as a national attention proxy.

    Raises:
        SourceError: the API is unreachable. The message explains the manual
            alternative rather than leaving the source silently empty.
    """
    start = cfg.data.start_date.strftime("%Y%m%d00")
    end = cfg.data.end_date.strftime("%Y%m%d00")

    for article in WIKIPEDIA_ARTICLES:
        url = _PAGEVIEWS_URL.format(article=article, start=start, end=end)
        request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                items = json.load(response)["items"]
        except (urllib.error.URLError, KeyError, json.JSONDecodeError):
            continue
        return pd.DataFrame(
            {
                "date": [pd.to_datetime(item["timestamp"][:8], format="%Y%m%d") for item in items],
                "value": [float(item["views"]) for item in items],
            }
        )

    raise SourceError(
        "[awareness] no backend produced data: no Trends CSV exports, trendspy not "
        f"installed, and the Wikipedia Pageviews API was unreachable.\n{_ACQUISITION}"
    )


def _broadcast_national(national: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Repeat one national series across every state under study.

    Explicit and deliberate: the resulting feature has no between-state variation,
    so it cannot help a pooled model separate states. Kept because the temporal
    signal is still real, and because a null result here is reportable.
    """
    states = [normalise_state(state) for state in cfg.data.states]
    return pd.concat(
        [national.assign(state=state) for state in states], ignore_index=True
    )