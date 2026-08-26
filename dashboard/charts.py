"""Chart builders. Pure functions returning matplotlib figures.

No Streamlit here on purpose: a figure builder that does not import the UI
framework can be tested without one, and these carry the design decisions worth
testing.

Every drawing choice follows from :mod:`dashboard.theme`. Nothing in this module
names a colour or a size.

The conventions applied throughout:

* **Direct labels over legends.** A legend makes a reader look away from the line
  and match a colour swatch. A label at the end of the line does not.
* **Intervals as a soft band.** Dashed bounds read as two more series; a filled
  band reads as one forecast with width, which is what it is.
* **No chartjunk.** Two spines, faint gridlines, no shadows, no 3-D, no borders
  around plot areas.
* **Zero baselines only where zero means something.** Forced onto a count axis,
  where zero cases is a real quantity; not onto a log axis, where it is not.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from dashboard import theme
from dashboard.geo import TILE_CODES

matplotlib.use("Agg")


def _figure(width: float, height: float) -> tuple[Figure, matplotlib.axes.Axes]:
    """A figure carrying the theme's matplotlib settings."""
    with matplotlib.rc_context(theme.mpl_rc()):
        figure = Figure(figsize=(width, height), dpi=110)
        axes = figure.add_subplot(111)
    for key, value in theme.mpl_rc().items():
        if key.startswith("axes.spines."):
            axes.spines[key.rsplit(".", 1)[-1]].set_visible(value)
    _apply(axes)
    return figure, axes


def _apply(axes: matplotlib.axes.Axes) -> None:
    """Apply theme settings that rc_context does not reach on an existing axes."""
    axes.set_facecolor(theme.SURFACE)
    axes.grid(True, color=theme.NEUTRAL[100], linewidth=1.0)
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(theme.BORDER)
    axes.tick_params(colors=theme.INK_FAINT, labelsize=theme.TYPE["small"])


def empty_figure(message: str, width: float = 6.0, height: float = 2.4) -> Figure:
    """A placeholder that says why a chart is missing.

    An honest empty state. A blank panel reads as a bug; a sentence reads as a
    fact about the data.
    """
    figure, axes = _figure(width, height)
    axes.set_axis_off()
    axes.text(
        0.5, 0.5, message,
        ha="center", va="center", wrap=True,
        color=theme.INK_FAINT, fontsize=theme.TYPE["small"],
        transform=axes.transAxes,
    )
    return figure


def forecast_chart(
    history: pd.DataFrame,
    forecast: pd.DataFrame,
    *,
    projection: pd.DataFrame | None = None,
    highlight: pd.Timestamp | None = None,
    width: float = 7.6,
    height: float = 2.6,
) -> Figure:
    """Observed series against the forecast, with the interval as a band.

    Takes the same ``projection`` argument as
    :func:`dashboard.plots.forecast_chart` so an exported figure shows exactly
    what was on screen. A report whose chart differs from the dashboard it was
    taken from is worse than no chart.

    Args:
        history: Columns ``date`` and ``actual``.
        forecast: Columns ``date``, ``predicted`` and optionally ``lower``/``upper``.
        projection: Forward curve, same columns plus ``mode`` marking each step
            ``direct`` or ``recursive``. Drawn dashed, with the recursive region
            shaded, because it is a different kind of claim from the fitted series.
        highlight: Period the rest of the page is showing, marked on the axis so
            the numbers in the panels can be located on the series.
        width: Figure width in inches.
        height: Figure height in inches.
    """
    has_projection = projection is not None and not projection.empty
    if history.empty and forecast.empty and not has_projection:
        return empty_figure("No forecast available for this state.", width, height)

    figure, axes = _figure(width, height)

    if highlight is not None:
        axes.axvline(highlight, color=theme.ACCENT, linewidth=1.2, zorder=2)

    if {"lower", "upper"} <= set(forecast.columns) and not forecast.empty:
        axes.fill_between(
            forecast["date"], forecast["lower"], forecast["upper"],
            color=theme.BAND, linewidth=0, zorder=1,
        )

    if not history.empty:
        axes.plot(
            history["date"], history["actual"],
            color=theme.INK_MUTED, zorder=3,
        )
        _label_line(
            axes, history["date"], history["actual"], "actual",
            theme.INK_MUTED, lift=-theme.SPACE["md"],
        )

    if not forecast.empty:
        axes.plot(
            forecast["date"], forecast["predicted"],
            color=theme.ACCENT, zorder=4,
        )
        _label_line(
            axes, forecast["date"], forecast["predicted"], "predicted",
            theme.ACCENT, lift=theme.SPACE["md"],
        )

    if has_projection:
        assert projection is not None
        _draw_projection(axes, projection)

    # Zero is a real quantity on a case-rate axis, so the baseline belongs there.
    axes.set_ylim(bottom=0)
    axes.set_ylabel("cases per 100k")
    figure.tight_layout()
    return figure


def _draw_projection(
    axes: matplotlib.axes.Axes, projection: pd.DataFrame
) -> None:
    """Draw the forward curve dashed, and shade where it turns recursive.

    The boundary is the single most important thing the chart says. A reader has
    to see where the model stopped using measured inputs and started reading its
    own output, and a continuous line would present the two as one.
    """
    if {"lower", "upper"} <= set(projection.columns):
        axes.fill_between(
            projection["date"], projection["lower"], projection["upper"],
            color=theme.BAND, alpha=0.55, linewidth=0, zorder=1,
        )

    recursive = (
        projection[projection["mode"] == "recursive"]
        if "mode" in projection.columns
        else projection.iloc[0:0]
    )
    if not recursive.empty:
        axes.axvspan(
            recursive["date"].min(), projection["date"].max(),
            color=theme.NEUTRAL[100], zorder=0,
        )
        axes.annotate(
            "recursive",
            xy=(recursive["date"].min(), 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(theme.SPACE["xs"], -theme.SPACE["sm"]),
            textcoords="offset points",
            ha="left", va="top",
            fontsize=theme.TYPE["small"], color=theme.INK_FAINT,
        )

    axes.plot(
        projection["date"], projection["predicted"],
        color=theme.ACCENT, linestyle=(0, (1, 1.6)), zorder=5,
    )
    _label_line(
        axes, projection["date"], projection["predicted"], "projected",
        theme.ACCENT, lift=theme.SPACE["lg"],
    )


def _label_line(
    axes: matplotlib.axes.Axes,
    x: Sequence[object],
    y: Sequence[float],
    label: str,
    colour: str,
    lift: int = 0,
) -> None:
    """Put the series name at the end of the line rather than in a legend.

    Args:
        axes: Axes to annotate.
        x: The series' x values.
        y: The series' y values.
        label: Text to place at the end of the line.
        colour: Label colour, matching the line.
        lift: Vertical offset in points. Two series that finish at nearly the same
            value would otherwise print their labels on top of each other, which
            is worse than the legend the direct label was meant to replace.
    """
    values = np.asarray(y, dtype=float)
    finite = np.flatnonzero(np.isfinite(values))
    if finite.size == 0:
        return
    last = finite[-1]
    axes.annotate(
        label,
        xy=(list(x)[last], values[last]),
        xytext=(theme.SPACE["xs"], lift),
        textcoords="offset points",
        va="center", ha="left",
        color=colour, fontsize=theme.TYPE["small"],
        fontweight=theme.WEIGHT["medium"],
    )


def attribution_chart(
    labels: Sequence[str],
    values: Sequence[float],
    *,
    width: float = 7.6,
    height: float = 2.8,
) -> Figure:
    """Horizontal bars of SHAP attribution, largest magnitude at the top.

    Horizontal because the labels are phrases — "rainfall lag-3" — and rotated
    vertical tick labels are a reliable way to make a chart unreadable.
    """
    if not len(labels):
        return empty_figure("No cached attributions for this state.", width, height)

    order = np.argsort(np.abs(np.asarray(values, dtype=float)))
    ordered_labels = [labels[position] for position in order]
    ordered_values = np.asarray(values, dtype=float)[order]

    figure, axes = _figure(width, height)
    positions = np.arange(len(ordered_values))
    axes.barh(
        positions, ordered_values,
        color=theme.ACCENT, height=0.62, zorder=3,
    )
    axes.set_yticks(positions)
    axes.set_yticklabels(ordered_labels, fontsize=theme.TYPE["small"])
    axes.axvline(0, color=theme.BORDER_STRONG, linewidth=1.0, zorder=2)
    axes.grid(axis="y", visible=False)
    axes.set_xlabel("contribution to the prediction")
    figure.tight_layout()
    return figure


def scenario_chart(
    baseline: pd.DataFrame,
    scenario: pd.DataFrame,
    *,
    width: float = 7.6,
    height: float = 2.6,
) -> Figure:
    """Baseline against scenario, both direct-labelled.

    The baseline is drawn in grey and the scenario in the accent, so the thing
    that changed is the thing that is coloured.
    """
    if baseline.empty:
        return empty_figure("Run a scenario to compare.", width, height)

    figure, axes = _figure(width, height)
    axes.plot(baseline["date"], baseline["predicted"], color=theme.INK_FAINT, zorder=3)
    _label_line(
        axes, baseline["date"], baseline["predicted"], "baseline",
        theme.INK_MUTED, lift=-theme.SPACE["md"],
    )

    if not scenario.empty:
        axes.plot(scenario["date"], scenario["predicted"], color=theme.ACCENT, zorder=4)
        _label_line(
            axes, scenario["date"], scenario["predicted"], "scenario",
            theme.ACCENT, lift=theme.SPACE["md"],
        )

    axes.set_ylim(bottom=0)
    axes.set_ylabel("cases per 100k")
    figure.tight_layout()
    return figure


def tile_map(
    values: Mapping[str, float],
    positions: Mapping[str, tuple[int, int]],
    breaks: list[float],
    *,
    selected: str | None = None,
    width: float = 6.4,
    height: float = 5.9,
) -> Figure:
    """A tile cartogram: one square per state, coloured by risk.

    A tile grid rather than true boundaries, because the repository carries no
    state geometry and fetching it at render time would make the dashboard depend
    on the network. The trade is deliberate and worth naming in the caption: a
    cartogram gives every state equal visual weight, which is arguably fairer for
    reading risk than a choropleth where Rajasthan shouts and Delhi vanishes.

    **Every state in the grid is drawn**, not only those carrying a forecast.
    States outside the study are filled in a neutral lighter than any risk band,
    so "not studied" cannot be misread as "low risk".

    Drawing the whole country is also what makes the panel fit. The twelve
    configured states span eight rows and five columns, which is a tall narrow
    shape that overflows a fixed-height panel and pushed the southern states --
    including, in the worst case, the selected one -- out of view. The full grid
    is close to square.

    Args:
        values: Risk value per state. States absent are drawn as no-data.
        positions: ``(row, column)`` grid position per state.
        breaks: Numeric band edges, so the legend can state them.
        selected: State to outline in the accent.
        width: Figure width in inches.
        height: Figure height in inches.
    """
    if not values:
        return empty_figure("No forecasts to map.", width, height)

    figure, axes = _figure(width, height)
    axes.set_axis_off()
    axes.grid(False)

    rows = [row for row, _ in positions.values()]
    columns = [column for _, column in positions.values()]

    for state, (row, column) in positions.items():
        has_forecast = state in values
        axes.add_patch(
            matplotlib.patches.Rectangle(
                (column, -row), 0.92, 0.92,
                facecolor=(
                    theme.risk_colour(values[state], breaks)
                    if has_forecast
                    else theme.NO_DATA
                ),
                hatch=None if has_forecast else theme.NO_DATA_HATCH,
                edgecolor=theme.ACCENT if state == selected else theme.TILE_BORDER,
                linewidth=2.0 if state == selected else 1.4,
                zorder=3,
            )
        )
        axes.text(
            column + 0.46, -row + 0.46, TILE_CODES.get(state, state[:2].upper()),
            ha="center", va="center",
            fontsize=theme.TYPE["small"],
            fontweight=theme.WEIGHT["medium"] if state == selected else theme.WEIGHT["regular"],
            color=theme.INK if has_forecast else theme.INK_FAINT,
            zorder=4,
        )

    axes.set_xlim(min(columns) - 0.2, max(columns) + 1.2)
    axes.set_ylim(-max(rows) - 0.2, -min(rows) + 1.2)
    axes.set_aspect("equal")
    figure.tight_layout()
    return figure


def legend_bands(breaks: list[float]) -> list[tuple[str, str]]:
    """Explicit numeric bands for the map legend, as ``(colour, label)`` pairs.

    Spelled out rather than shown as a smooth gradient bar, so a reader can say
    what any tile means rather than estimating from a strip.
    """
    labels: list[tuple[str, str]] = []
    previous = 0.0
    for position, edge in enumerate(breaks):
        labels.append((theme.RISK_RAMP[position], f"{previous:.2f} to {edge:.2f}"))
        previous = edge
    labels.append((theme.RISK_RAMP[len(breaks)], f"{previous:.2f} and above"))
    labels.append((theme.NO_DATA, "not in this study"))
    return labels


