"""Artifact loading. The dashboard's only route to data.

Everything here is a read. Forecasts, recommendations, thresholds and SHAP values
are all computed by ``scripts/build_dashboard_data.py`` and stored through
:mod:`src.artifacts`; this module fetches them and nothing more.

The one exception is scenario simulation, which cannot be precomputed because the
scenario is chosen at render time. That path is explicit, lives in
:func:`run_scenario`, and is the only place the dashboard does real work.

Loading is cached for the session. A dashboard that re-reads a Parquet file on
every widget interaction feels broken even when it is correct.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import streamlit as st

from src.artifacts import load_run
from src.config import Config, load_config

#: Columns every watchlist carries, empty or not. An empty frame that dropped
#: them would make the panel raise on a sort rather than render its empty state.
WATCHLIST_COLUMNS = ("state", "predicted", "threshold", "exceedance")

#: Run name written by ``scripts/build_dashboard_data.py``.
DASHBOARD_RUN = "dashboard"


class DashboardDataError(RuntimeError):
    """Raised when the dashboard cannot find what it needs to render."""


@dataclass(frozen=True)
class DashboardData:
    """Everything the interface reads, loaded once.

    Attributes:
        forecasts: Point forecast and interval per state and origin date.
        recommendations: Tier, trigger value and actions per forecast row.
        thresholds: Per-state tier boundaries with the evidence behind each.
        history: Observed case rate per state and date.
        panel: The cleaned panel, kept so scenarios can be run against it.
        meta: Which model produced this, and when.
    """

    forecasts: pd.DataFrame
    recommendations: pd.DataFrame
    thresholds: pd.DataFrame
    history: pd.DataFrame
    panel: pd.DataFrame
    meta: dict[str, Any]

    @property
    def states(self) -> list[str]:
        """States with at least one forecast, in display order."""
        return sorted(self.forecasts["state"].unique())

    @property
    def target_periods(self) -> list[pd.Timestamp]:
        """Every period a forecast is available for, oldest first.

        These are *target* periods, not forecast origins: the question a reader
        asks is "what is predicted for October?", not "what did we predict from
        September?".
        """
        if self.forecasts.empty:
            return []
        return sorted(pd.to_datetime(self.forecasts["target_date"]).unique())

    def at_period(self, period: pd.Timestamp) -> pd.DataFrame:
        """Every state's forecast for one target period, indexed by state.

        What the map draws. Falls back to the nearest earlier period when a state
        has no forecast for exactly this one, so a state never silently vanishes
        from the map because its series is a month shorter.
        """
        if self.forecasts.empty:
            return self.forecasts
        upto = self.forecasts[pd.to_datetime(self.forecasts["target_date"]) <= period]
        if upto.empty:
            return self.latest_by_state()
        return upto.sort_values("target_date").groupby("state").tail(1).set_index("state")

    def latest_by_state(self) -> pd.DataFrame:
        """The most recent forecast per state."""
        if self.forecasts.empty:
            return self.forecasts
        newest = self.forecasts.sort_values("origin_date").groupby("state").tail(1)
        return newest.set_index("state")

    def for_state(self, state: str) -> pd.DataFrame:
        """Every forecast for one state, oldest first."""
        return self.forecasts[self.forecasts["state"] == state].sort_values("origin_date")

    def history_for(self, state: str) -> pd.DataFrame:
        """Observed series for one state, oldest first."""
        return self.history[self.history["state"] == state].sort_values("date")

    def recommendation_for(
        self, state: str, period: pd.Timestamp | None = None
    ) -> pd.Series | None:
        """The recommendation for one state, at a period or at the latest one."""
        rows = self.recommendations[self.recommendations["state"] == state]
        if rows.empty:
            return None
        rows = rows.assign(_target=pd.to_datetime(rows["target_date"])).sort_values("_target")
        if period is not None:
            upto = rows[rows["_target"] <= period]
            if not upto.empty:
                rows = upto
        return rows.iloc[-1]

    def thresholds_for(self, state: str) -> pd.DataFrame:
        """Tier boundaries for one state."""
        return self.thresholds[self.thresholds["state"] == state]


@st.cache_resource(show_spinner=False)
def production() -> Any:
    """The frozen production model, loaded once per session.

    ``cache_resource`` rather than ``cache_data``: a Keras model is a live object,
    not a value to be copied. Without this the model is rebuilt on **every**
    widget interaction -- a slider drag, a ticked checkbox -- because Streamlit
    re-runs the whole script each time and nothing underneath holds it.
    """
    from src.production import load_production

    return load_production()


@st.cache_data(show_spinner=False)
def load() -> DashboardData:
    """Read the precomputed dashboard artifact, once per session.

    Cached because Streamlit re-runs the entire script on every interaction, and
    a dashboard that re-reads six Parquet files to redraw one checkbox feels
    broken even when it is correct.

    Raises:
        DashboardDataError: the artifact has not been built yet.
    """
    try:
        payload = load_run(DASHBOARD_RUN)
    except FileNotFoundError as exc:
        raise DashboardDataError(
            "no dashboard artifact yet. Run:\n\n"
            "    python scripts/build_dashboard_data.py --synthetic\n\n"
            "The dashboard reads precomputed results; it does not compute them."
        ) from exc

    required = {"forecasts", "recommendations", "thresholds", "history", "panel", "meta"}
    missing = sorted(required - set(payload))
    if missing:
        raise DashboardDataError(
            f"dashboard artifact is missing {missing}; rebuild it with "
            "scripts/build_dashboard_data.py"
        )

    return DashboardData(
        forecasts=payload["forecasts"],
        recommendations=payload["recommendations"],
        thresholds=payload["thresholds"],
        history=payload["history"],
        panel=_restore_panel(payload["panel"]),
        meta=payload["meta"],
    )


@st.cache_data(show_spinner=False)
def load_attributions(state: str) -> tuple[list[str], list[float]]:
    """SHAP attributions for one state, averaged over its explained rows.

    Cached per state. The values are read from an artifact and never change
    within a session, so recomputing the mean on every rerun buys nothing.

    Returns empty lists when nothing was cached for that state, which is normal:
    attributions are computed for a bounded sample of rows, not every one.
    """
    from src.explain import load_attribution

    try:
        attribution = load_attribution()
        spec = production().spec
    except Exception:  # noqa: BLE001 - an absent cache is a normal state, not an error
        return [], []

    rows = attribution.sample_index.get_level_values("state") == state
    if not rows.any():
        return [], []

    frame = attribution.frame()[rows].mean()
    drivers = [
        (column, float(value))
        for column, value in frame.items()
        if spec.origins[column].raw_variable not in {"__state__"}
    ]
    drivers.sort(key=lambda item: abs(item[1]), reverse=True)
    top = drivers[:8]

    from src.explain import describe_column

    return [describe_column(spec.origins[column]) for column, _ in top], [
        value for _, value in top
    ]


def run_scenario(
    panel: pd.DataFrame,
    variable: str,
    change: float,
    mode: str,
    cfg: Config,
) -> Any:
    """The one live computation the dashboard performs.

    Rebuilds the feature pipeline against a modified panel, which is why the
    caller shows a spinner. Everything else on screen was read from disk.
    """
    from src.simulate import Scenario, simulate

    return simulate(
        panel,
        Scenario(variable=variable, change=change, mode=mode),
        production(),
        cfg,
    )


@st.cache_data(show_spinner=False)
def forecast_curve(state: str, horizon: int) -> pd.DataFrame:
    """Project one state forward, cached on the selection that asked for it.

    The second live computation in the dashboard, and the reason it is cached
    rather than precomputed: the horizon is chosen at render time, and a recursive
    projection re-runs the whole feature pipeline once per step. Keyed on
    :meth:`~dashboard.selection.Selection.key`, so it runs once when the selection
    changes and not at all on a re-render.

    Args:
        state: Area to project.
        horizon: Periods past the trained horizon to reach for. ``0`` returns
            nothing, which is the fitted-series-only case.

    Returns:
        The curve as a frame with ``target_date``, ``predicted``, ``lower``,
        ``upper``, ``mode`` and ``reliability``; empty when nothing was asked for
        or the state cannot be projected.
    """
    if horizon <= 0:
        return pd.DataFrame(
            columns=["target_date", "steps_ahead", "predicted", "lower", "upper",
                     "mode", "reliability"]
        )

    from src.simulate import SimulationError, forecast_horizon

    dataset = load()
    cfg = load_config()
    try:
        model = production()
        last = dataset.panel.loc[state, cfg.data.target_column].dropna().index.max()
        curve = forecast_horizon(
            dataset.panel,
            state,
            pd.Timestamp(last) + _period_offset(cfg) * horizon,
            model,
            cfg,
        )
    except (SimulationError, FileNotFoundError, KeyError):
        # A state with too little history, or no frozen model. Both are states the
        # interface has to render rather than crash on.
        return pd.DataFrame(
            columns=["target_date", "steps_ahead", "predicted", "lower", "upper",
                     "mode", "reliability"]
        )
    return curve.frame()


def _period_offset(cfg: Config) -> pd.DateOffset:
    """One period at the configured granularity."""
    if cfg.project.granularity == "monthly":
        return pd.DateOffset(months=1)
    return pd.DateOffset(weeks=1)


def watchlist(dataset: DashboardData, period: pd.Timestamp) -> pd.DataFrame:
    """States whose forecast crosses their own high tier at or before a period.

    Ranked by how far past the threshold they are, in units of the threshold, so
    a small state and a large one are comparable. A raw case rate would sort the
    list by population density instead of by concern.

    Returns:
        :data:`WATCHLIST_COLUMNS`,
        highest first. Empty when nothing crosses, which is a real answer and
        should be shown as one.
    """
    at_period = dataset.at_period(period)
    if at_period.empty or dataset.thresholds.empty:
        return pd.DataFrame(columns=WATCHLIST_COLUMNS)

    high = dataset.thresholds[dataset.thresholds["tier"] == _top_tier(dataset)]
    bounds = dict(zip(high["state"], high["value_cases_per_100k"], strict=True))

    rows = []
    for state, row in at_period.iterrows():
        limit = bounds.get(state)
        value = row["upper_cases_per_100k"]
        if limit is None or pd.isna(limit) or limit <= 0 or value < limit:
            continue
        rows.append(
            {
                "state": state,
                "predicted": float(value),
                "threshold": float(limit),
                "exceedance": float(value / limit - 1.0),
            }
        )
    if not rows:
        # An empty watchlist is the common case and a real answer. Building the
        # frame from an empty list would lose the columns and make sorting raise.
        return pd.DataFrame(columns=WATCHLIST_COLUMNS)
    return pd.DataFrame(rows).sort_values("exceedance", ascending=False)


def _top_tier(dataset: DashboardData) -> str:
    """The most severe tier name present in the thresholds table."""
    tiers = list(dataset.thresholds["tier"].unique())
    return tiers[-1] if tiers else "HIGH"


def config() -> Config:
    """The loaded configuration, for tier names and interval coverage."""
    return load_config()


def _restore_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Put the panel back on its ``(state, date)`` index after a Parquet round trip."""
    if isinstance(panel.index, pd.MultiIndex):
        return panel
    return panel.set_index(["state", "date"]).sort_index()
