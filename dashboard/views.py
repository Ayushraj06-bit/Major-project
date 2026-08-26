"""The views. Each reads from :mod:`dashboard.data` and renders components.

No view computes anything, with two documented exceptions — the scenario panel
and the forward projection — because neither a what-if nor a user-chosen horizon
can be precomputed. Both go through cached functions in :mod:`dashboard.data`, so
they run once per selection change rather than once per render, and both say on
screen that they are live.

**Every view takes the same :class:`~dashboard.selection.Selection`.** No view
holds a copy of the state or the period, and no view has a control that changes
what another view is describing. A control that moved only the map would leave
the panels beside it answering a different question.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from dashboard import charts, components, data, plots, theme
from dashboard.geo import TILE_POSITIONS
from dashboard.selection import KEY_STATE, Selection, select_state

#: How far behind "now" the data may fall before the chart says so.
#:
#: A projection runs forward from the last observation, not from today. Roughly a
#: season of drift is where "forward projection" stops meaning "what happens next"
#: and starts needing a caveat.
STALE_AFTER_DAYS = 120


def risk_breaks(frame: pd.DataFrame) -> list[float]:
    """Four numeric band edges from the observed spread of predicted risk.

    Quantiles of what is actually on the map, so the bands separate the states
    being displayed rather than describing some other distribution. The legend
    prints them, so a reader can say what any tile means.
    """
    values = frame["upper_cases_per_100k"].dropna()
    if values.empty:
        return [0.25, 0.5, 0.75, 1.0]
    return [float(values.quantile(q)) for q in (0.2, 0.4, 0.6, 0.8)]


def map_view(dataset: data.DashboardData, selection: Selection) -> None:
    """National risk map for the selected period. Clicking a tile selects it.

    Recolours as the period changes, because the values it draws come from
    :meth:`~dashboard.data.DashboardData.at_period` and nothing here caches a
    frame of its own.
    """
    at_period = dataset.at_period(selection.period)
    if at_period.empty:
        components.empty_state(
            "No forecasts to map.",
            "Rebuild with scripts/build_dashboard_data.py.",
        )
        return

    breaks = risk_breaks(at_period)
    values = at_period["upper_cases_per_100k"].to_dict()

    # Legend above the map, not below it. The panel has a fixed height, so a
    # legend underneath sat past the scroll fold, and a legend nobody scrolls to
    # is a map nobody can read.
    components.legend(
        # From the static chart module rather than restated here: the map on
        # screen and the map in the report must agree about what a colour means.
        charts.legend_bands(breaks),
        caption=(
            f"Predicted cases per 100,000 for "
            f"{selection.period.strftime('%B %Y')}, at the "
            f"{dataset.meta['interval_coverage']:.0%} interval upper bound. "
            "One equal tile per state. Click a tile to select it."
        ),
    )

    event = st.plotly_chart(
        plots.tile_map(
            values, TILE_POSITIONS, breaks,
            selected=selection.state,
            height=theme.PANEL_HEIGHT["map"] - theme.CHART_INSET["map"],
        ),
        use_container_width=True,
        config=plots.TOOLBAR,
        on_select="rerun",
        key="map_click",
    )
    _apply_map_click(event, dataset.states)


def _apply_map_click(event: object, states: list[str]) -> None:
    """Route a tile click into the selection store.

    Guarded rather than trusted: the event payload is Streamlit's, its shape has
    changed between versions, and a click on empty space returns a selection with
    no points in it.
    """
    points = getattr(event, "selection", {}) or {}
    chosen = [
        point.get("customdata")
        for point in (points.get("points") or [])
        if point.get("customdata")
    ]
    if not chosen:
        return
    name = chosen[0][0] if isinstance(chosen[0], list) else chosen[0]
    if name in states and name != st.session_state.get(KEY_STATE):
        select_state(str(name))
        st.rerun()


def summary_view(dataset: data.DashboardData, selection: Selection) -> None:
    """Headline numbers for the selected state and period."""
    if dataset.for_state(selection.state).empty:
        components.empty_state(
            f"No forecast for {selection.state}.",
            "This state produced no complete feature window. Its neighbours may "
            "lie outside the configured study area, or its series may be too short.",
        )
        return

    at_period = dataset.at_period(selection.period)
    if selection.state not in at_period.index:
        components.empty_state(
            f"No forecast for {selection.state} in "
            f"{selection.period.strftime('%B %Y')}.",
            "Choose another period from the rail.",
        )
        return

    row = at_period.loc[selection.state]
    components.metric_row(
        [
            (
                "Predicted",
                f"{row['predicted_cases_per_100k']:.2f}",
                "cases per 100,000",
            ),
            (
                f"{dataset.meta['interval_coverage']:.0%} interval",
                f"{row['lower_cases_per_100k']:.2f}–{row['upper_cases_per_100k']:.2f}",
                "lower to upper bound",
            ),
            (
                "For",
                pd.Timestamp(row["target_date"]).strftime("%b %Y"),
                f"from origin {pd.Timestamp(row['origin_date']).strftime('%b %Y')}",
            ),
        ]
    )


def forecast_view(dataset: data.DashboardData, selection: Selection) -> None:
    """Observed series, fitted forecast, and the forward projection if asked for.

    The projection is the one place the interface shows the model reading its own
    output. Where that starts is drawn on the chart and stated underneath, because
    a recursive step and a direct forecast are different kinds of claim and the
    reader is entitled to know which one they are looking at.
    """
    history = dataset.history_for(selection.state)
    forecasts = dataset.for_state(selection.state)

    if history.empty and forecasts.empty:
        components.empty_state(f"No series recorded for {selection.state}.")
        return

    projection = _projection_frame(selection)
    figure = plots.forecast_chart(
        history=history[["date", "actual"]],
        forecast=_forecast_frame(forecasts, selection.show_uncertainty),
        projection=projection,
        highlight=selection.period,
        height=theme.PANEL_HEIGHT["chart"] - theme.CHART_INSET["plain"],
    )
    st.plotly_chart(figure, use_container_width=True, config=plots.TOOLBAR)
    components.note(_forecast_caption(selection, projection))


def _forecast_frame(forecasts: pd.DataFrame, show_uncertainty: bool) -> pd.DataFrame:
    """The fitted series, shaped for the chart and stripped of bands if hidden."""
    frame = forecasts.rename(
        columns={
            "target_date": "date",
            "predicted_cases_per_100k": "predicted",
            "lower_cases_per_100k": "lower",
            "upper_cases_per_100k": "upper",
        }
    )
    columns = ["date", "predicted"] + (["lower", "upper"] if show_uncertainty else [])
    return frame[columns]


def _projection_frame(selection: Selection) -> pd.DataFrame:
    """The forward curve for this selection, shaped for the chart.

    Empty when no projection was asked for, or when the state cannot be projected.
    Both are normal and both render as simply no dashed line.
    """
    with components.loading(
        f"Projecting {selection.state} {selection.horizon} periods ahead…"
    ):
        curve = data.forecast_curve(selection.state, selection.horizon)
    if curve.empty:
        return curve
    frame = curve.rename(columns={"target_date": "date"})
    columns = ["date", "predicted", "mode"] + (
        ["lower", "upper"] if selection.show_uncertainty else []
    )
    return frame[columns]


def _forecast_caption(selection: Selection, projection: pd.DataFrame) -> str:
    """What the chart is showing, including what it cannot promise."""
    base = (
        f"Whole series for {selection.state}. The vertical rule marks "
        f"{selection.period.strftime('%B %Y')}, the period shown above."
    )
    if projection.empty:
        if selection.projecting:
            return base + (
                f" No forward projection available for {selection.state}: it has "
                "too little history to project from."
            )
        return base

    origin = pd.Timestamp(projection["date"].min()) - pd.DateOffset(months=1)
    base += _staleness_note(origin)

    recursive = projection[projection["mode"] == "recursive"]
    if recursive.empty:
        return base + " The dotted line is a direct forecast from observed inputs."

    start = pd.Timestamp(recursive["date"].min()).strftime("%B %Y")
    return base + (
        f" The dotted line is a forward projection. From {start} it is "
        "<b>recursive</b>: the model is reading its own earlier predictions, with "
        "climatological normals standing in for weather nobody has observed. The "
        "band widens with every step, and that widening is an acknowledgement "
        "rather than a coverage guarantee."
    )


def _staleness_note(last_observed: pd.Timestamp) -> str:
    """Say what the projection is measured from, when that is not roughly now.

    The projection runs forward from the panel's last observation, which is not
    the same thing as running forward from today. With data ending in December
    2023, a "six months ahead" curve reaches mid-2024 -- comfortably in the past.
    Drawn without this note it reads as a forecast of the coming season, which is
    the single most damaging way to misread this chart.
    """
    behind = (pd.Timestamp.today().normalize() - last_observed).days
    if behind < STALE_AFTER_DAYS:
        return ""
    return (
        f" <b>Projected forward from {last_observed.strftime('%B %Y')}</b>, the "
        f"last period with data — about {behind // 30} months ago. These are not "
        "predictions about the coming season."
    )


def export_view(dataset: data.DashboardData, selection: Selection) -> None:
    """Offer the current forecast chart as a PNG and a PDF.

    Built from :mod:`dashboard.charts`, the same matplotlib code the report uses,
    so what a reader downloads is what the write-up will contain. The Plotly
    figure on screen is for interaction; this is for paper.
    """
    history = dataset.history_for(selection.state)
    forecasts = dataset.for_state(selection.state)
    if history.empty and forecasts.empty:
        components.empty_state("Nothing to export for this state.")
        return

    figure = charts.forecast_chart(
        history=history[["date", "actual"]],
        forecast=_forecast_frame(forecasts, selection.show_uncertainty),
        projection=_projection_frame(selection),
        highlight=selection.period,
    )
    components.download_figure(
        figure,
        f"{selection.state.lower().replace(' ', '-')}-"
        f"{selection.period.strftime('%Y-%m')}",
    )
    components.note(
        "Vector PDF or 200 dpi PNG, drawn by the same code as the report figures."
    )


def comparison_view(dataset: data.DashboardData, selection: Selection) -> None:
    """The selected state against the others chosen for comparison."""
    if len(selection.overlay) < 2:
        components.empty_state(
            "Choose states to compare in the rail.",
            "The selected state is drawn in the accent; the rest stay grey.",
        )
        return

    series = {
        name: _comparison_series(dataset, name) for name in selection.overlay
    }
    st.plotly_chart(
        plots.comparison_chart(
            series, focus=selection.state,
            height=theme.PANEL_HEIGHT["chart"] - theme.CHART_INSET["plain"],
        ),
        use_container_width=True,
        config=plots.TOOLBAR,
    )
    components.note(
        "Point forecasts only. Overlapping interval bands say nothing, and "
        "comparison is about relative level and shape."
    )


def _comparison_series(dataset: data.DashboardData, state: str) -> pd.DataFrame:
    """One state's fitted forecast, shaped for the comparison chart."""
    rows = dataset.for_state(state)
    if rows.empty:
        return pd.DataFrame(columns=["date", "predicted"])
    return pd.DataFrame(
        {"date": rows["target_date"], "predicted": rows["predicted_cases_per_100k"]}
    )


def watchlist_view(dataset: data.DashboardData, selection: Selection) -> None:
    """States crossing their own high threshold at the selected period.

    Keyboard reachable: each row is a button, so the list can be walked with Tab
    and entered with Return rather than requiring a pointer on a map tile.
    """
    rows = data.watchlist(dataset, selection.period)
    if rows.empty:
        components.empty_state(
            f"No state crosses its high threshold in "
            f"{selection.period.strftime('%B %Y')}.",
            "An empty watchlist is a result, not a missing panel.",
        )
        return

    for _, row in rows.iterrows():
        left, right = st.columns([3, 2], gap="small")
        with left:
            if st.button(
                str(row["state"]),
                key=f"watch_{row['state']}",
                use_container_width=True,
            ):
                select_state(str(row["state"]))
                st.rerun()
        with right:
            components.note(
                f"{row['predicted']:.2f} vs {row['threshold']:.2f} "
                f"(+{row['exceedance']:.0%})"
            )
    components.note(
        "Ranked by exceedance of each state's own threshold, so a small state and "
        "a large one are comparable."
    )


def attribution_view(selection: Selection) -> None:
    """Cached SHAP attribution for the selected state."""
    labels, values = data.load_attributions(selection.state)
    if not labels:
        components.empty_state(
            f"No cached attributions for {selection.state}.",
            "Attributions are computed for a bounded sample of rows. Run "
            "scripts/run_shap.py to refresh the cache.",
        )
        return

    st.plotly_chart(
        plots.attribution_chart(
            labels, values,
            height=theme.PANEL_HEIGHT["chart"] - theme.CHART_INSET["plain"],
        ),
        use_container_width=True,
        config=plots.TOOLBAR,
    )
    components.note(
        "Mean SHAP contribution across this state's explained forecasts, summed "
        "over the input window. Read as association, not cause."
    )


def scenario_view(dataset: data.DashboardData, selection: Selection) -> None:
    """Scenario controls and the baseline comparison.

    A what-if, not a forecast. It answers "what if conditions were different?",
    while the projection on the forecast chart answers "what happens next under
    typical conditions?" — and the two must not be read as one thing.

    Its controls change only this panel, which is why they live here rather than
    in the rail: a scenario is a question about the selected state, not a change
    to what the page is showing.
    """
    panel_columns = [
        column
        for column in dataset.panel.columns
        if column not in {"population", "population_density"}
    ]
    control, result = st.columns([1, 2], gap="large")

    with control:
        variable = st.selectbox("Variable", panel_columns, key="scenario_variable")
        mode = st.radio("Change", ["percent", "absolute"], key="scenario_mode",
                        horizontal=True)
        change = st.slider(
            "Amount", min_value=-50.0, max_value=50.0, value=20.0, step=5.0,
            key="scenario_change",
        )
        run = st.button("Run scenario", use_container_width=True)

    with result:
        if not run:
            components.empty_state(
                "Choose a variable and an amount, then run.",
                "Scenarios rebuild every derived feature, so this takes a moment.",
            )
            return

        with st.spinner("Rebuilding features and re-predicting…"):
            outcome = data.run_scenario(
                dataset.panel, variable, change, mode, data.config()
            )

        baseline = _series(outcome.baseline, selection.state)
        scenario = _series(outcome.scenario_forecast, selection.state)
        st.plotly_chart(
            plots.scenario_chart(
                baseline, scenario,
                height=theme.PANEL_HEIGHT["chart"] - theme.CHART_INSET["controls"],
            ),
            use_container_width=True,
            config=plots.TOOLBAR,
        )

        components.evidence_list(
            [
                ("Scenario", outcome.scenario.describe()),
                ("Mean change", f"{outcome.mean_delta:+.3f} cases per 100,000"),
                (
                    "Clamped",
                    f"{outcome.clamped_rows} of {outcome.modified_rows} cells "
                    f"({outcome.clamped_fraction:.0%})",
                ),
            ]
        )
        if outcome.out_of_distribution:
            components.note(
                "Some values were held to the observed range. Beyond it the model "
                "has no evidence to draw on."
            )
        if not outcome.affects_model:
            components.note(
                f"{variable} feeds no model input, so a zero change is the model "
                "ignoring it rather than the variable not mattering."
            )
        components.note(
            "The model learns correlation, not causation. This shows how the model "
            "responds, not how transmission responds."
        )


def _series(frame: pd.DataFrame, state: str) -> pd.DataFrame:
    """One state's forecast series, shaped for the scenario chart."""
    rows = frame[frame["state"] == state].sort_values("origin_date")
    if rows.empty:
        return pd.DataFrame(columns=["date", "predicted"])
    return pd.DataFrame(
        {
            "date": rows["target_date"],
            "predicted": rows.get(
                "predicted_cases_per_100k", np.expm1(rows["predicted_log"])
            ),
        }
    )


def recommendation_view(dataset: data.DashboardData, selection: Selection) -> None:
    """The recommendation card: tier, the numbers behind it, and the actions."""
    recommendation = dataset.recommendation_for(selection.state, selection.period)
    if recommendation is None:
        components.empty_state(
            f"No recommendation for {selection.state}.",
            "A recommendation needs both a forecast and a derived threshold.",
        )
        return

    st.markdown(components.tier_badge(recommendation["tier"]), unsafe_allow_html=True)
    components.spacer("sm")

    threshold = recommendation["threshold_value_cases_per_100k"]
    components.evidence_list(
        [
            (
                "Triggered by",
                f"{recommendation['trigger_value_cases_per_100k']:.2f} cases per "
                f"100,000 ({recommendation['trigger_basis']})",
            ),
            (
                "Compared against",
                (
                    f"{threshold:.2f}, the {recommendation['threshold_label']} for "
                    f"{selection.state} "
                    f"(n={recommendation['threshold_observations']:.0f})"
                    if pd.notna(threshold)
                    else recommendation["threshold_label"]
                ),
            ),
            (
                "Top drivers",
                ", ".join(recommendation["top_drivers"]) or "not cached for this period",
            ),
        ]
    )

    components.spacer("md")
    components.panel_header("Recommended actions")
    actions = data.config().risk.actions_for(recommendation["tier"])
    st.markdown(
        '<div class="evidence">'
        + "".join(f"<div>— {action}</div>" for action in actions)
        + "</div>",
        unsafe_allow_html=True,
    )
    components.note(f"Action catalogue: {dataset.meta['action_source']}")


def thresholds_view(dataset: data.DashboardData, selection: Selection) -> None:
    """The tier boundaries this state is judged against.

    Shown beside the recommendation because "above the 90th percentile" only
    means something next to the number it refers to, and because it fills the
    detail column with evidence rather than whitespace.
    """
    thresholds = dataset.thresholds_for(selection.state)
    if thresholds.empty:
        components.empty_state(f"No thresholds derived for {selection.state}.")
        return

    components.evidence_list(
        [
            (
                f"{row['tier']} at",
                f"{row['value_cases_per_100k']:.2f} cases per 100,000 — "
                f"{row['label']}, from {int(row['n_observations'])} observations",
            )
            for _, row in thresholds.iterrows()
        ]
    )
