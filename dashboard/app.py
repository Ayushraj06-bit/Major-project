"""Dengue outbreak intelligence dashboard.

Layout only. Every number shown here was computed by
``scripts/build_dashboard_data.py`` and read from the artifact store; the two
exceptions are the forward projection and scenario simulation, both marked as
live in the interface.

The page answers one question at a time: **what is predicted for this area, for
this period?** Both are chosen in the left rail, held in
:class:`~dashboard.selection.Selection`, and every panel follows them. No panel
keeps a copy and no panel has a control that changes what another panel is
describing.

Run with::

    streamlit run dashboard/app.py
"""

from __future__ import annotations

import time

import streamlit as st

from dashboard import components, data, selection, theme, views


def main() -> None:
    """Compose the page: a left rail for controls, a canvas for content."""
    st.set_page_config(
        page_title="Dengue outbreak intelligence",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    components.inject_theme()

    try:
        dataset = data.load()
    except data.DashboardDataError as error:
        components.page_header("Dengue outbreak intelligence")
        components.empty_state(str(error).split("\n")[0], _hint(error))
        return

    if not dataset.states:
        components.page_header("Dengue outbreak intelligence")
        components.empty_state("The artifact contains no states.")
        return

    current = rail(dataset)
    components.page_header("Dengue outbreak intelligence", _provenance(dataset, current))
    _canvas(dataset, current)

    if current.playing:
        # The period itself is advanced at the top of `rail`, before the slider
        # bound to that key exists. All this does is pace the loop and ask for
        # the next frame; advancing here would write a widget key too late and
        # raise, which is precisely what pressing Play used to do.
        time.sleep(selection.PLAY_INTERVAL_SECONDS)
        st.rerun()


def _canvas(dataset: data.DashboardData, current: selection.Selection) -> None:
    """The panels, in reading order: where, what, why, what-if."""
    left, right = st.columns([1, 1], gap="large")

    with left, components.panel(
        f"National risk · {current.period.strftime('%b %Y')}",
        height=theme.PANEL_HEIGHT["map"],
    ):
        views.map_view(dataset, current)

    with right, components.panel(current.state, height=theme.PANEL_HEIGHT["detail"]):
        views.summary_view(dataset, current)
        components.spacer("md")
        views.recommendation_view(dataset, current)
        components.spacer("md")
        components.panel_header("Thresholds for this state")
        views.thresholds_view(dataset, current)

    with components.panel("Observed and predicted", height=theme.PANEL_HEIGHT["chart"]):
        views.forecast_view(dataset, current)

    middle_left, middle_right = st.columns([1, 1], gap="large")
    with middle_left, components.panel("Compare states", height=theme.PANEL_HEIGHT["chart"]):
        views.comparison_view(dataset, current)
    with middle_right, components.panel(
        f"Watchlist · {current.period.strftime('%b %Y')}",
        height=theme.PANEL_HEIGHT["chart"],
    ):
        views.watchlist_view(dataset, current)

    lower_left, lower_right = st.columns([1, 1], gap="large")
    with lower_left, components.panel("Attribution", height=theme.PANEL_HEIGHT["chart"]):
        views.attribution_view(current)
    with lower_right, components.panel("Scenario", height=theme.PANEL_HEIGHT["chart"]):
        views.scenario_view(dataset, current)


def rail(dataset: data.DashboardData) -> selection.Selection:
    """The left rail: what to look at, when, and how far ahead. Then provenance.

    Draws the widgets and returns the resulting selection. The widgets write into
    :mod:`dashboard.selection`'s keys and nothing else reads them directly, so a
    panel cannot reach past this function to change what the page is showing.
    """
    periods = dataset.target_periods
    labels = [selection.period_label(period) for period in periods]

    with st.sidebar:
        # Both of these write widget-backed keys, so they have to happen before a
        # single widget below is instantiated. Streamlit raises otherwise.
        selection.apply_pending(dataset.states)
        if labels:
            st.session_state.setdefault(selection.KEY_PERIOD, labels[-1])
            if st.session_state.get(selection.KEY_PLAYING):
                selection.advance_period(periods)

        components.panel_header("Area")
        focus = str(
            st.selectbox(
                "Area", dataset.states, label_visibility="collapsed",
                key=selection.KEY_STATE,
            )
        )

        components.spacer("md")
        components.panel_header("Period")
        if labels:
            st.select_slider(
                "Period", options=labels, label_visibility="collapsed",
                key=selection.KEY_PERIOD,
            )
            st.button(
                "Pause" if st.session_state.get(selection.KEY_PLAYING) else "Play",
                on_click=selection.toggle_play,
                use_container_width=True,
                help="Step through every period, one per refresh.",
            )
        else:
            components.note("No periods in the artifact.")

        components.spacer("md")
        components.panel_header("Forward projection")
        st.session_state.setdefault(
            selection.KEY_HORIZON, selection.DEFAULT_HORIZON_MODE
        )
        st.radio(
            "Forward projection",
            list(selection.HORIZON_MODES),
            label_visibility="collapsed",
            key=selection.KEY_HORIZON,
        )
        components.note(
            "Past the trained horizon the model reads its own output. Those steps "
            "are drawn dotted and shaded, and the interval widens with each one."
        )

        components.spacer("md")
        components.panel_header("Display")
        st.checkbox(
            "Show prediction intervals", value=True, key=selection.KEY_UNCERTAINTY
        )
        st.multiselect(
            "Compare with",
            selection.comparison_options(dataset.states, focus),
            key=selection.KEY_COMPARE,
        )

        components.spacer("lg")
        components.panel_header("Export")
        current = selection.read(dataset.states, periods)
        views.export_view(dataset, current)

        components.spacer("lg")
        components.panel_header("Model")
        components.evidence_list(
            [
                ("Configuration", str(dataset.meta["experiment"])),
                ("Horizon", f"{dataset.meta['horizon']} period ahead"),
                ("Features", str(dataset.meta["n_features"])),
                ("Thresholds", str(dataset.meta["threshold_method"])),
            ]
        )
        if dataset.meta.get("synthetic"):
            components.spacer("md")
            components.note(
                "Synthetic data. Every figure here describes a generator, not dengue."
            )
    return current


def _provenance(
    dataset: data.DashboardData, current: selection.Selection
) -> str:
    """One line naming what produced these numbers, and what they are about."""
    line = (
        f"Showing {current.period.strftime('%B %Y')} at the "
        f"{dataset.meta['interval_coverage']:.0%} interval upper bound. "
        f"Frozen model {dataset.meta['experiment']}, trained "
        f"{str(dataset.meta['trained_at'])[:10]}."
    )
    if current.projecting:
        line += (
            f" Projecting {current.horizon} periods past the last observation; "
            "steps beyond the trained horizon are recursive."
        )
    return line


def _hint(error: Exception) -> str:
    """The remaining lines of a data error, rendered as guidance."""
    lines = str(error).split("\n")[1:]
    return "<br>".join(line for line in lines if line.strip())


if __name__ == "__main__":
    main()
