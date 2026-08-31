"""Design tokens. Every colour, size and spacing value in the dashboard lives here.

Nothing else in ``dashboard/`` may write a hex code or a pixel value. That is not
tidiness for its own sake: it is what makes the whole interface adjustable from
one place, and what stops a dashboard drifting into fifteen slightly different
greys as panels get added.

The design is deliberately restrained.

**One accent, spent carefully.** :data:`ACCENT` marks interactive elements and the
current selection, and nothing else. Everything structural is grey. A reader
should be able to find what is clickable and what is selected without reading a
word, and if the accent were removed entirely the hierarchy should still hold —
because it is carried by size and spacing, not colour.

**One typeface, three sizes, two weights.** Hierarchy comes from scale and
whitespace. Bold and colour are the tools you reach for when the layout is not
doing enough work.

**A 4px spacing scale.** Every gap is a multiple of four. Dense dashboards read as
unfinished, so the defaults here are generous.

**A sequential, colourblind-safe risk ramp.** ColorBrewer YlOrRd. Never a rainbow:
a rainbow ramp implies an ordering that its lightness does not follow, so it
misreads in greyscale and for the ~8% of men with red-green colour deficiency.
"""

from __future__ import annotations

from typing import Any

# --------------------------------------------------------------------------- #
# Colour
# --------------------------------------------------------------------------- #

#: The single accent. Interactive elements and current selection only.
ACCENT = "#0F6E8C"
ACCENT_STRONG = "#0A4E64"
ACCENT_WASH = "#E8F1F4"

#: Neutral scale, light to dark. Everything structural is drawn from this.
#:
#: The two text greys are set so that all three ink levels clear WCAG AA (4.5:1)
#: on the page background. The obvious lighter greys do not: a comfortable-looking
#: #A6A5A0 measures 2.47:1, which is unreadable for a good number of people and
#: looks fine to everyone else, so it has to be checked rather than eyeballed.
NEUTRAL: dict[int, str] = {
    0: "#FFFFFF",
    50: "#FAFAF9",
    100: "#F4F4F2",
    200: "#E6E5E2",
    300: "#D4D3CF",
    400: "#75746D",
    500: "#5D5C56",
    600: "#57564F",
    700: "#403F39",
    900: "#1B1A17",
}

#: Semantic roles, so views never index the neutral scale directly.
INK = NEUTRAL[900]
INK_MUTED = NEUTRAL[500]
INK_FAINT = NEUTRAL[400]
SURFACE = NEUTRAL[0]
SURFACE_SUNK = NEUTRAL[50]
BORDER = NEUTRAL[200]
BORDER_STRONG = NEUTRAL[300]

#: ColourBrewer YlOrRd, 5-class. Sequential and colourblind-safe.
RISK_RAMP: tuple[str, ...] = ("#FFFFB2", "#FECC5C", "#FD8D3C", "#F03B20", "#BD0026")

#: Discrete tier colours, drawn from the same ramp so map and badges agree.
TIER_COLOURS: dict[str, str] = {
    "LOW": RISK_RAMP[0],
    "MEDIUM": RISK_RAMP[2],
    "HIGH": RISK_RAMP[4],
}

#: Tier text colours, chosen for contrast against the fills above.
TIER_INK: dict[str, str] = {"LOW": NEUTRAL[700], "MEDIUM": NEUTRAL[900], "HIGH": "#FFFFFF"}

#: Prediction interval band. Soft, so the line reads first.
BAND = "#DCE7EB"

#: The seasonal profile's line and band.
#:
#: Drawn in muted ink rather than in a second accent colour. The accent means
#: "this is the forecast model"; a climatological pattern is a different and
#: weaker claim, and giving it equal visual weight would say otherwise. One
#: accent, spent on one thing.
SEASONAL_LINE = NEUTRAL[500]
SEASONAL_BAND = NEUTRAL[200]

#: Fill for a state carrying no forecast.
#:
#: Distinguished by hatching rather than by colour, which is the cartographic
#: convention and the only version that works. Luminance cannot do the job: the
#: lightest risk band is #FFFFB2 at 0.96, brighter than any grey that still reads
#: as a fill, so a plain grey tile would sit *between* the risk bands and could be
#: read as low risk. A hatch is unambiguous in greyscale and under any colour
#: vision.
NO_DATA = NEUTRAL[50]
NO_DATA_HATCH = "///"

#: Muted borders for map tiles. Black outlines make a map look like a diagram.
TILE_BORDER = NEUTRAL[0]

# --------------------------------------------------------------------------- #
# Type
# --------------------------------------------------------------------------- #

FONT_STACK = (
    '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", '
    "Arial, sans-serif"
)

#: Three sizes. Adding a fourth is almost always a sign the layout needs work.
TYPE: dict[str, int] = {"small": 12, "body": 15, "title": 22}

#: Two weights.
WEIGHT: dict[str, int] = {"regular": 400, "medium": 600}

# --------------------------------------------------------------------------- #
# Space
# --------------------------------------------------------------------------- #

#: 4px scale. Every gap in the interface is one of these.
SPACE: dict[str, int] = {"xs": 4, "sm": 8, "md": 16, "lg": 24, "xl": 32}

#: Fixed panel heights, so switching states does not reflow the page.
#:
#: Sized to the tallest content each panel holds, verified by screenshot rather
#: than by arithmetic. A fixed-height container in Streamlit scrolls rather than
#: grows, so a value that is merely close clips the last row of the map without
#: any error to notice. Map and detail match so the two columns stay aligned.
PANEL_HEIGHT: dict[str, int] = {"map": 700, "chart": 340, "detail": 700}

#: Vertical space a panel spends before a chart gets any, by panel kind.
#:
#: A figure height is its panel's height minus its chrome. Writing that
#: subtraction with a bare number in a view is how the layout starts drifting:
#: the next person adds a caption, nudges the number, and the two panels beside
#: it no longer line up. The values are measured from screenshots, not derived --
#: Streamlit's own header and caption heights are not published.
CHART_INSET: dict[str, int] = {
    "plain": 60,      # panel header, plus the caption under the chart
    "controls": 140,  # ... plus a row of controls sharing the panel
    "map": 200,       # ... plus the legend, which sits above the map
}

RADIUS = 4
BORDER_WIDTH = 1

#: Right-hand gutter reserved for a direct label at the end of a line.
#:
#: The charts label series at the end of the line instead of in a legend, which
#: only works if the label has somewhere to go. With the default margin the word
#: is drawn past the plotting area and clipped by the panel, which is worse than
#: the legend it replaced.
LABEL_GUTTER = 96

#: Periods visible by default on a time series, before the reader pans or zooms.
#:
#: Not the whole series. Twelve years of monthly history compresses a six-month
#: forward projection into a few pixels, so the one part of the chart a reader
#: came for becomes the one part they cannot see. Three years is enough seasonal
#: context to judge a forecast against, and zooming out is one gesture away.
DEFAULT_WINDOW_PERIODS = 36

# --------------------------------------------------------------------------- #
# Motion
# --------------------------------------------------------------------------- #

#: One duration and one easing for everything that moves.
#:
#: Two transition speeds on one page read as a bug rather than as a style. 220ms
#: is long enough to be seen as motion and short enough not to be waited on;
#: ease-out because a movement that decelerates into place feels settled, while
#: one that accelerates out of it feels thrown.
TRANSITION_MS = 220
EASING = "cubic-in-out"
EASING_CSS = "cubic-bezier(0.4, 0.0, 0.2, 1)"

# --------------------------------------------------------------------------- #
# Derived stylesheets
# --------------------------------------------------------------------------- #


def streamlit_config() -> dict[str, str]:
    """Streamlit's own ``[theme]`` settings, derived from the tokens above.

    Streamlit styles its widgets from its own theme, not from our stylesheet. With
    no theme configured it follows the **viewer's OS colour-scheme preference** --
    so on a machine set to dark mode it drew near-white label text and near-black
    input boxes on top of the white page :func:`css` pins. Labels came out white on
    white at about 1.05:1, which is not low contrast, it is invisible.

    Pinning the base theme is also what stops Streamlit spending its default red on
    every slider, radio and checkbox. This project has one accent and it is
    :data:`ACCENT`; a second one arriving through the widget library is still a
    second one.

    The values are returned rather than written into ``.streamlit/config.toml`` by
    hand, so that file stays checkable against these tokens instead of drifting
    from them.
    """
    return {
        "base": "light",
        "primaryColor": ACCENT,
        "backgroundColor": SURFACE,
        "secondaryBackgroundColor": SURFACE_SUNK,
        "textColor": INK,
    }


def css() -> str:
    """The stylesheet, built from the tokens above.

    Every value here is interpolated from a token. A literal in this string would
    be the same mistake as a literal in a view.
    """
    return f"""
<style>
  html, body, [class*="css"] {{
    font-family: {FONT_STACK};
    color: {INK};
  }}
  .stApp {{ background: {SURFACE}; }}

  /* Streamlit's own chrome competes with the content. */
  #MainMenu, footer, header {{ visibility: hidden; }}
  .block-container {{
    padding-top: {SPACE["lg"]}px;
    padding-bottom: {SPACE["xl"]}px;
    max-width: 1400px;
  }}

  h1, h2, h3 {{ font-weight: {WEIGHT["medium"]}; color: {INK}; letter-spacing: -0.01em; }}
  h1 {{ font-size: {TYPE["title"]}px; margin: 0 0 {SPACE["xs"]}px 0; }}

  .panel {{
    border: {BORDER_WIDTH}px solid {BORDER};
    border-radius: {RADIUS}px;
    background: {SURFACE};
    padding: {SPACE["md"]}px;
    margin-bottom: {SPACE["md"]}px;
  }}
  .panel-header {{
    font-size: {TYPE["small"]}px;
    font-weight: {WEIGHT["medium"]};
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: {INK_MUTED};
    padding-bottom: {SPACE["sm"]}px;
    margin-bottom: {SPACE["md"]}px;
    border-bottom: {BORDER_WIDTH}px solid {BORDER};
  }}
  .panel-note {{
    font-size: {TYPE["small"]}px;
    color: {INK_FAINT};
    margin-top: {SPACE["sm"]}px;
  }}

  .tile-label {{
    font-size: {TYPE["small"]}px;
    color: {INK_MUTED};
    margin-bottom: {SPACE["xs"]}px;
  }}
  .tile-value {{
    font-size: {TYPE["title"]}px;
    font-weight: {WEIGHT["medium"]};
    color: {INK};
    line-height: 1.1;
  }}
  .tile-context {{
    font-size: {TYPE["small"]}px;
    color: {INK_FAINT};
    margin-top: {SPACE["xs"]}px;
  }}

  .badge {{
    display: inline-block;
    padding: {SPACE["xs"]}px {SPACE["sm"]}px;
    border-radius: {RADIUS}px;
    font-size: {TYPE["small"]}px;
    font-weight: {WEIGHT["medium"]};
    letter-spacing: 0.04em;
  }}

  .empty {{
    display: flex;
    align-items: center;
    justify-content: center;
    text-align: center;
    color: {INK_FAINT};
    font-size: {TYPE["small"]}px;
    background: {SURFACE_SUNK};
    border: {BORDER_WIDTH}px dashed {BORDER_STRONG};
    border-radius: {RADIUS}px;
    padding: {SPACE["lg"]}px;
  }}

  .evidence {{
    font-size: {TYPE["small"]}px;
    color: {INK_MUTED};
    line-height: 1.7;
  }}
  .evidence b {{ font-weight: {WEIGHT["medium"]}; color: {INK}; }}

  /* The accent, spent only on interaction and selection. */
  .stButton > button {{
    background: {ACCENT};
    color: {SURFACE};
    border: none;
    border-radius: {RADIUS}px;
    font-weight: {WEIGHT["medium"]};
    font-size: {TYPE["small"]}px;
    padding: {SPACE["sm"]}px {SPACE["md"]}px;
  }}
  .stButton > button:hover {{ background: {ACCENT_STRONG}; color: {SURFACE}; }}
  .selected {{
    border-left: {SPACE["xs"]}px solid {ACCENT};
    background: {ACCENT_WASH};
    padding-left: {SPACE["sm"]}px;
  }}
  section[data-testid="stSidebar"] {{
    background: {SURFACE_SUNK};
    border-right: {BORDER_WIDTH}px solid {BORDER};
  }}
  section[data-testid="stSidebar"] .block-container {{ padding-top: {SPACE["lg"]}px; }}
</style>
"""


def mpl_rc() -> dict[str, Any]:
    """Matplotlib settings, derived from the same tokens.

    Charts drawn with these match the surrounding interface without any view
    setting a colour of its own.
    """
    return {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": BORDER,
        "axes.labelcolor": INK_MUTED,
        "axes.labelsize": TYPE["small"],
        "axes.titlesize": TYPE["small"],
        "axes.titlecolor": INK_MUTED,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": NEUTRAL[100],
        "grid.linewidth": 1.0,
        "xtick.color": INK_FAINT,
        "ytick.color": INK_FAINT,
        "xtick.labelsize": TYPE["small"],
        "ytick.labelsize": TYPE["small"],
        "text.color": INK,
        "font.size": TYPE["small"],
        "legend.frameon": False,
        # Hatch colour is global in matplotlib; a patch cannot set it without
        # also changing its edge, which the selection outline needs.
        "hatch.color": BORDER_STRONG,
        "hatch.linewidth": 0.8,
        "lines.linewidth": 1.6,
        "lines.solid_capstyle": "round",
        # No chartjunk: two spines carry an axis perfectly well.
        "axes.spines.top": False,
        "axes.spines.right": False,
    }


def plotly_template() -> dict[str, Any]:
    """The Plotly layout template, from the same tokens as everything else.

    Interactive charts have to look like the static ones or the page reads as two
    designs stitched together. This is the same set of decisions as
    :func:`mpl_rc` -- two spines, faint gridlines, muted ink, no chartjunk --
    expressed in the second library's vocabulary. Neither function names a colour.
    """
    axis = {
        "showgrid": True,
        "gridcolor": NEUTRAL[100],
        "gridwidth": 1,
        "zeroline": False,
        "linecolor": BORDER,
        "tickcolor": BORDER,
        "tickfont": {"size": TYPE["small"], "color": INK_FAINT},
        "title": {"font": {"size": TYPE["small"], "color": INK_MUTED}},
        "showspikes": False,
    }
    return {
        "layout": {
            "paper_bgcolor": SURFACE,
            "plot_bgcolor": SURFACE,
            "font": {"family": FONT_STACK, "size": TYPE["small"], "color": INK},
            "colorway": [ACCENT, INK_FAINT, RISK_RAMP[3], ACCENT_STRONG, RISK_RAMP[1]],
            "xaxis": axis,
            "yaxis": axis,
            "margin": {"l": SPACE["xl"] + SPACE["md"], "r": LABEL_GUTTER,
                       "t": SPACE["md"], "b": SPACE["xl"]},
            "hoverlabel": {
                "bgcolor": NEUTRAL[900],
                "bordercolor": NEUTRAL[900],
                "font": {"family": FONT_STACK, "size": TYPE["small"], "color": SURFACE},
            },
            "hovermode": "x unified",
            "showlegend": False,
            "transition": {"duration": TRANSITION_MS, "easing": EASING},
            "dragmode": "pan",
        }
    }


def swatch_background(colour: str) -> str:
    """CSS background for a legend swatch of this colour.

    Hatched when the colour is :data:`NO_DATA`, matching the map. The pairing of
    that fill with a hatch is a single rule and lives here, so a legend cannot
    drift into showing a plain grey square for a tile the map draws hatched --
    which would put the two back into the ambiguity the hatch exists to remove.
    """
    if colour != NO_DATA:
        return colour
    return (
        f"repeating-linear-gradient(45deg, {NO_DATA}, {NO_DATA} 2px, "
        f"{BORDER_STRONG} 2px, {BORDER_STRONG} 3px)"
    )


def risk_colour(value: float, breaks: list[float]) -> str:
    """Colour for a risk value, given explicit numeric breaks.

    Explicit breaks rather than a continuous normalisation, so the legend can
    state exactly what each band means. A reader who cannot say what a colour
    stands for is looking at decoration.
    """
    for position, edge in enumerate(breaks):
        if value < edge:
            return RISK_RAMP[min(position, len(RISK_RAMP) - 1)]
    return RISK_RAMP[-1]
