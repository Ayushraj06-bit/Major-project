"""Interactive chart builders. Pure functions returning Plotly figures.

The counterpart to :mod:`dashboard.charts`, and the split is deliberate rather
than duplication:

* :mod:`dashboard.charts` draws **matplotlib** figures for export — the report,
  the PNG, the PDF. Vector output, no runtime, reproducible from a script.
* This module draws **Plotly** figures for the **screen**, where hover, pan, zoom
  and click are the point. A static image cannot do any of them.

Both read :mod:`dashboard.theme`, so the two libraries produce one visual
language. Nothing here names a colour or a size.

The conventions from the static charts carry over unchanged — direct labels
instead of legends, intervals as a soft band, no chartjunk, a zero baseline only
where zero is a real quantity — with two additions that only apply on screen:

* **Hover is the legend.** A unified hover box gives every series' value at one
  x position, which is what a reader was going to compare anyway.
* **Recursive projection is drawn differently from direct forecast.** A dashed
  line and a separate band, because the two are different kinds of claim and a
  single continuous line would present them as one.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd
import plotly.graph_objects as go

from dashboard import theme
from dashboard.geo import TILE_CODES

#: Plotly's toolbar, trimmed to what is useful here.
TOOLBAR: dict[str, object] = {
    "displaylogo": False,
    "modeBarButtonsToRemove": [
        "select2d", "lasso2d", "autoScale2d", "toggleSpikelines",
        "hoverClosestCartesian", "hoverCompareCartesian",
    ],
    "scrollZoom": True,
    "displayModeBar": "hover",
}

#: Hover template for a case-rate series. One place, so every chart agrees.
_RATE_HOVER = "%{y:.2f} per 100k"


def _figure(height: int) -> go.Figure:
    """An empty figure carrying the theme template."""
    figure = go.Figure()
    figure.update_layout(theme.plotly_template()["layout"], height=height)
    return figure


def empty_figure(message: str, height: int = 260) -> go.Figure:
    """A placeholder that says why a chart is missing.

    The interactive twin of :func:`dashboard.charts.empty_figure`. A blank panel
    reads as a bug; a sentence reads as a fact about the data.
    """
    figure = _figure(height)
    figure.add_annotation(
        text=message, x=0.5, y=0.5, xref="paper", yref="paper",
        showarrow=False, align="center",
        font={"size": theme.TYPE["small"], "color": theme.INK_FAINT},
    )
    figure.update_layout(
        xaxis={"visible": False}, yaxis={"visible": False}, hovermode=False
    )
    return figure


def _band(
    figure: go.Figure,
    frame: pd.DataFrame,
    colour: str,
    name: str,
    *,
    opacity: float = 1.0,
) -> None:
    """Add an interval as a filled band between ``lower`` and ``upper``.

    Drawn as one shape rather than two dashed bounds: a band reads as a single
    forecast with width, which is what a prediction interval is. Two lines read as
    two more series.
    """
    if frame.empty or not {"lower", "upper"} <= set(frame.columns):
        return
    figure.add_trace(
        go.Scatter(
            x=list(frame["date"]) + list(frame["date"])[::-1],
            y=list(frame["upper"]) + list(frame["lower"])[::-1],
            fill="toself", fillcolor=colour, opacity=opacity, mode="lines",
            # The outline is given the band's own colour rather than only zero
            # width: a hairline still renders at some device pixel ratios, and it
            # showed up as a row of dots tracing the interval bounds.
            line={"width": 0, "color": colour},
            hoverinfo="skip", showlegend=False, name=name,
        )
    )


def _line(
    figure: go.Figure,
    frame: pd.DataFrame,
    colour: str,
    name: str,
    *,
    dash: str | None = None,
    width: float | None = None,
) -> None:
    """Add a series with a hover readout and a direct label at its end."""
    if frame.empty:
        return
    figure.add_trace(
        go.Scatter(
            x=frame["date"], y=frame["predicted"], mode="lines", name=name,
            line={
                "color": colour,
                "width": width if width is not None else 2.0,
                "dash": dash or "solid",
                "shape": "linear",
            },
            hovertemplate=_RATE_HOVER + "<extra>" + name + "</extra>",
        )
    )


def _direct_label(
    figure: go.Figure, frame: pd.DataFrame, colour: str, text: str, shift: int = 0
) -> None:
    """Put the series name at the end of the line rather than in a legend.

    Args:
        figure: Figure to annotate.
        frame: The series, with ``date`` and ``predicted``.
        colour: Label colour, matching the line.
        text: Text to place at the end of the line.
        shift: Vertical offset in pixels. Two series finishing at nearly the same
            value would otherwise print their labels on top of each other, which
            is worse than the legend the direct label replaced.
    """
    if frame.empty:
        return
    last = frame.iloc[-1]
    figure.add_annotation(
        x=last["date"], y=last["predicted"], text=text,
        showarrow=False, xanchor="left", xshift=theme.SPACE["xs"], yshift=shift,
        font={"size": theme.TYPE["small"], "color": colour,
              "weight": theme.WEIGHT["medium"]},
    )


def forecast_chart(
    history: pd.DataFrame,
    forecast: pd.DataFrame,
    *,
    projection: pd.DataFrame | None = None,
    highlight: pd.Timestamp | None = None,
    height: int = 300,
) -> go.Figure:
    """Observed series, fitted forecast, and the forward projection beyond it.

    The forward projection is drawn as a **dashed** line with its own band, and
    the boundary between direct and recursive is marked on the axis. The reader
    has to be able to see where the model stopped using measured inputs and
    started reading its own output; a single continuous line would hide it.

    Args:
        history: Columns ``date`` and ``actual``.
        forecast: Columns ``date``, ``predicted`` and optionally ``lower``/``upper``.
        projection: Forward curve, same columns plus ``mode`` marking each step
            ``direct`` or ``recursive``.
        highlight: Period the rest of the page is showing, marked on the axis.
        height: Figure height in pixels.
    """
    has_projection = projection is not None and not projection.empty
    if history.empty and forecast.empty and not has_projection:
        return empty_figure("No forecast available for this state.", height)

    figure = _figure(height)

    _band(figure, forecast, theme.BAND, "interval")
    if has_projection:
        assert projection is not None
        _band(figure, projection, theme.BAND, "projection interval", opacity=0.55)

    if not history.empty:
        figure.add_trace(
            go.Scatter(
                x=history["date"], y=history["actual"], mode="lines", name="observed",
                line={"color": theme.INK_FAINT, "width": 1.6},
                hovertemplate=_RATE_HOVER + "<extra>observed</extra>",
            )
        )
        _direct_label(
            figure,
            history.rename(columns={"actual": "predicted"}),
            theme.INK_MUTED, "observed", -theme.SPACE["md"],
        )

    _line(figure, forecast, theme.ACCENT, "predicted")
    _direct_label(figure, forecast, theme.ACCENT, "predicted", theme.SPACE["md"])

    if has_projection:
        assert projection is not None
        _line(
            figure, _bridged(forecast, projection), theme.ACCENT, "projected",
            dash="dot", width=2.2,
        )
        _direct_label(
            figure, projection, theme.ACCENT, "projected", theme.SPACE["md"]
        )
        _mark_projection_start(figure, projection)

    if highlight is not None:
        figure.add_vline(
            x=highlight, line={"color": theme.ACCENT, "width": 1.2},
        )

    # Zero is a real quantity on a case-rate axis, so the baseline belongs there.
    figure.update_yaxes(rangemode="tozero", title_text="cases per 100k")
    _focus_recent(figure, history, forecast, projection)
    return figure


def _focus_recent(figure: go.Figure, *frames: pd.DataFrame | None) -> None:
    """Open on the recent window rather than the whole series.

    Twelve years of monthly history compresses a six-month projection into a few
    pixels, and the projection is the part a reader came for. Panning and zooming
    are enabled, so the rest of the series is one gesture away; it just is not
    what the chart opens on.

    Applied to every time series on the page, not only the forecast. Two charts
    side by side showing different spans is worse than either span alone, because
    a reader compares their shapes without checking their axes.
    """
    dates = [
        frame["date"] for frame in frames if frame is not None and not frame.empty
    ]
    if not dates:
        return
    combined = pd.to_datetime(pd.concat(dates)).drop_duplicates().sort_values()
    if len(combined) <= theme.DEFAULT_WINDOW_PERIODS:
        return
    figure.update_xaxes(
        range=[combined.iloc[-theme.DEFAULT_WINDOW_PERIODS], combined.iloc[-1]]
    )


def _bridged(forecast: pd.DataFrame, projection: pd.DataFrame) -> pd.DataFrame:
    """Join the projection line back to the last fitted point.

    The two are consecutive periods drawn as separate traces, which leaves a gap
    the width of one period between them. A gap in a time series means missing
    data, and this one means nothing at all -- so the line is joined while the
    dashes still mark where the fitted series ends and the projection begins.
    """
    if forecast.empty:
        return projection
    tail = forecast.iloc[[-1]][["date", "predicted"]]
    return pd.concat([tail, projection[["date", "predicted"]]], ignore_index=True)


def _mark_projection_start(figure: go.Figure, projection: pd.DataFrame) -> None:
    """Shade the recursive region and name it.

    The single most important thing this chart says. A projection where the model
    is reading its own output has to look different from one built on measured
    inputs, and a caption alone does not survive someone glancing at the shape.
    """
    if "mode" not in projection.columns:
        return
    recursive = projection[projection["mode"] == "recursive"]
    if recursive.empty:
        return

    start = pd.Timestamp(recursive["date"].min())
    figure.add_vrect(
        x0=start, x1=pd.Timestamp(projection["date"].max()),
        fillcolor=theme.NEUTRAL[100], opacity=0.7, line_width=0, layer="below",
        annotation_text="recursive", annotation_position="top left",
        annotation_font={"size": theme.TYPE["small"], "color": theme.INK_FAINT},
    )


def comparison_chart(
    series: Mapping[str, pd.DataFrame],
    *,
    focus: str | None = None,
    height: int = 300,
) -> go.Figure:
    """Several states' forecasts on one axis, the focused one in the accent.

    The same builder as :func:`forecast_chart` would need a second interval band
    per state, and five overlapping translucent bands say nothing. Comparison is
    about relative level and shape, so the lines carry it alone and the hover box
    reports every state's value at whatever period the cursor is over.

    Args:
        series: State name to a frame with ``date`` and ``predicted``.
        focus: The selected state, drawn in the accent while the rest stay grey.
        height: Figure height in pixels.
    """
    populated = {name: frame for name, frame in series.items() if not frame.empty}
    if not populated:
        return empty_figure("Choose states to compare.", height)

    figure = _figure(height)
    for name, frame in populated.items():
        is_focus = name == focus
        _line(
            figure, frame,
            theme.ACCENT if is_focus else theme.NEUTRAL[300],
            name,
            width=2.4 if is_focus else 1.4,
        )
    for name, frame in populated.items():
        _direct_label(
            figure, frame,
            theme.ACCENT if name == focus else theme.INK_FAINT,
            name,
        )

    figure.update_yaxes(rangemode="tozero", title_text="cases per 100k")
    _focus_recent(figure, *populated.values())
    return figure


def attribution_chart(
    labels: Sequence[str], values: Sequence[float], *, height: int = 300
) -> go.Figure:
    """Horizontal SHAP bars, largest magnitude at the top.

    Horizontal because the labels are phrases — "rainfall lag-3" — and rotated
    vertical tick labels are a reliable way to make a chart unreadable.
    """
    if not len(labels):
        return empty_figure("No cached attributions for this state.", height)

    ordered = sorted(zip(labels, values, strict=True), key=lambda pair: abs(pair[1]))
    figure = _figure(height)
    figure.add_trace(
        go.Bar(
            x=[value for _, value in ordered],
            y=[label for label, _ in ordered],
            orientation="h",
            marker={"color": theme.ACCENT},
            hovertemplate="%{y}: %{x:+.4f}<extra></extra>",
        )
    )
    figure.add_vline(x=0, line={"color": theme.BORDER_STRONG, "width": 1})
    figure.update_layout(hovermode="closest", bargap=0.38)
    figure.update_xaxes(title_text="contribution to the prediction")
    figure.update_yaxes(showgrid=False)
    return figure


def scenario_chart(
    baseline: pd.DataFrame, scenario: pd.DataFrame, *, height: int = 300
) -> go.Figure:
    """Baseline against scenario, both direct-labelled.

    The baseline is grey and the scenario is the accent, so the thing that
    changed is the thing that is coloured.
    """
    if baseline.empty:
        return empty_figure("Run a scenario to compare.", height)

    figure = _figure(height)
    _line(figure, baseline, theme.INK_FAINT, "baseline", width=1.6)
    _line(figure, scenario, theme.ACCENT, "scenario", width=2.2)
    _direct_label(figure, baseline, theme.INK_MUTED, "baseline", -theme.SPACE["md"])
    _direct_label(figure, scenario, theme.ACCENT, "scenario", theme.SPACE["md"])

    figure.update_yaxes(rangemode="tozero", title_text="cases per 100k")
    _focus_recent(figure, baseline, scenario)
    return figure


#: Diagonal offsets, as a fraction of a tile, for the no-data hatch.
_HATCH_OFFSETS = (-0.6, -0.3, 0.0, 0.3, 0.6)


def _hatch(
    figure: go.Figure,
    states: Sequence[str],
    positions: Mapping[str, tuple[int, int]],
) -> None:
    """Rule diagonal lines across the tiles that carry no forecast.

    Drawn as line segments rather than as a shape fill pattern, which this
    Plotly version does not offer on shapes. One trace holds every segment, with
    a gap between each, so the whole hatch costs one trace regardless of how many
    states are outside the study.
    """
    if not states:
        return

    xs: list[float | None] = []
    ys: list[float | None] = []
    for state in states:
        row, column = positions[state]
        x0, y0, side = column + 0.04, -row + 0.04, 0.92
        for fraction in _HATCH_OFFSETS:
            offset = fraction * side
            start, end = max(0.0, offset), min(side, side + offset)
            xs += [x0 + start, x0 + end, None]
            ys += [y0 + start - offset, y0 + end - offset, None]

    figure.add_trace(
        go.Scatter(
            x=xs, y=ys, mode="lines",
            line={"color": theme.BORDER_STRONG, "width": 1},
            hoverinfo="skip", showlegend=False,
        )
    )


def tile_map(
    values: Mapping[str, float],
    positions: Mapping[str, tuple[int, int]],
    breaks: list[float],
    *,
    selected: str | None = None,
    height: int = 520,
) -> go.Figure:
    """A clickable tile cartogram: one square per state, coloured by risk.

    The interactive twin of :func:`dashboard.charts.tile_map`, and it keeps that
    function's two decisions intact:

    * **Every state in the grid is drawn**, not only those carrying a forecast.
      States outside the study are filled in a neutral and hatched, so "not
      studied" cannot be misread as "low risk" — luminance alone cannot separate
      them, because the lightest risk band is brighter than any usable grey.
    * A cartogram rather than true boundaries, so every state gets equal visual
      weight and the panel does not depend on fetching geometry at render time.

    What it adds is hover and click. Each tile reports its state and value on
    hover, and returns its state name through Streamlit's selection events.
    """
    if not values:
        return empty_figure("No forecasts to map.", height)

    figure = _figure(height)
    states = list(positions)
    rows = [positions[state][0] for state in states]
    columns = [positions[state][1] for state in states]

    # Tiles as shapes rather than as marker fills, so a state with no forecast can
    # be hatched. That distinction is not decorative: the lightest risk band is
    # brighter than any usable grey, so a plain grey tile sits *between* the risk
    # bands and reads as low risk. A hatch is unambiguous in greyscale and under
    # any colour vision, and it is the cartographic convention.
    #
    # Assigned in one update rather than through 36 `add_shape` calls. Each call
    # revalidates the whole layout, so appending them one at a time is quadratic
    # and cost more than every other panel on the page put together.
    figure.update_layout(
        shapes=[
            {
                "type": "rect",
                "x0": positions[state][1] + 0.04,
                "x1": positions[state][1] + 0.96,
                "y0": -positions[state][0] + 0.04,
                "y1": -positions[state][0] + 0.96,
                "fillcolor": (
                    theme.risk_colour(values[state], breaks)
                    if state in values
                    else theme.NO_DATA
                ),
                "line": {
                    "color": (
                        theme.ACCENT if state == selected else theme.TILE_BORDER
                    ),
                    "width": 3 if state == selected else 1.5,
                },
                "layer": "below",
            }
            for state in states
        ]
    )

    _hatch(figure, [state for state in states if state not in values], positions)

    # A transparent point per tile, carrying the label, the hover and the click.
    figure.add_trace(
        go.Scatter(
            x=[column + 0.5 for column in columns],
            y=[-row + 0.5 for row in rows],
            mode="markers+text",
            marker={"symbol": "square", "size": 34, "opacity": 0},
            text=[TILE_CODES.get(state, state[:2].upper()) for state in states],
            textfont={
                "size": theme.TYPE["small"],
                "color": [
                    theme.INK if state in values else theme.INK_FAINT
                    for state in states
                ],
            },
            customdata=states,
            hovertemplate="<b>%{customdata}</b><br>%{meta}<extra></extra>",
            meta=[
                f"{values[state]:.2f} per 100k"
                if state in values
                else "not in this study"
                for state in states
            ],
        )
    )

    figure.update_layout(
        hovermode="closest",
        clickmode="event+select",
        dragmode=False,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
    )
    figure.update_xaxes(
        visible=False, range=[min(columns) - 0.2, max(columns) + 1.2],
    )
    figure.update_yaxes(
        visible=False, range=[-max(rows) - 0.2, -min(rows) + 1.2],
        scaleanchor="x", scaleratio=1,
    )
    return figure
