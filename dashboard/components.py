"""The reusable pieces. One implementation per repeated element.

Every panel, tile, header and empty state in the interface comes from here. That
is what keeps the spacing consistent and stops the fourth panel someone adds from
having its own idea about padding.

None of these writes a colour or a size: they read :mod:`dashboard.theme`.
"""

from __future__ import annotations

import io
from collections.abc import Iterator, Sequence
from contextlib import contextmanager

import streamlit as st
from matplotlib.figure import Figure

from dashboard import theme

#: Export formats offered, as extension to MIME type.
EXPORT_FORMATS: dict[str, str] = {"png": "image/png", "pdf": "application/pdf"}


def inject_theme() -> None:
    """Apply the stylesheet. Called once, at the top of the app."""
    st.markdown(theme.css(), unsafe_allow_html=True)


def page_header(title: str, subtitle: str = "") -> None:
    """The one page title. Size and spacing carry it, not colour."""
    st.markdown(f"<h1>{title}</h1>", unsafe_allow_html=True)
    if subtitle:
        st.markdown(
            f'<div class="panel-note">{subtitle}</div>', unsafe_allow_html=True
        )
    spacer("md")


def panel_header(label: str) -> None:
    """A section label: small, uppercase, muted, with a hairline rule under it."""
    st.markdown(f'<div class="panel-header">{label}</div>', unsafe_allow_html=True)


@contextmanager
def panel(label: str, height: int) -> Iterator[None]:
    """A bordered region with a header and a fixed height.

    The height is required rather than optional, because a panel without one is
    how a page starts reflowing: switching from a state with twelve
    recommendations to one with two makes the whole layout jump, and a page that
    moves under the cursor feels unreliable even when the numbers are right.
    Heights come from :data:`~dashboard.theme.PANEL_HEIGHT`.
    """
    with st.container(height=height, border=True):
        panel_header(label)
        yield


def metric_tile(label: str, value: str, context: str = "") -> None:
    """A single number with its label above and its context beneath.

    The value is the largest thing in the tile; the label and context sit at the
    small size in muted ink. No colour is used to distinguish them.
    """
    st.markdown(
        f'<div class="tile-label">{label}</div>'
        f'<div class="tile-value">{value}</div>'
        + (f'<div class="tile-context">{context}</div>' if context else ""),
        unsafe_allow_html=True,
    )


def metric_row(tiles: Sequence[tuple[str, str, str]]) -> None:
    """A row of metric tiles on equal columns."""
    columns = st.columns(len(tiles), gap="medium")
    for column, (label, value, context) in zip(columns, tiles, strict=True):
        with column:
            metric_tile(label, value, context)


def tier_badge(tier: str) -> str:
    """A tier chip, coloured from the same ramp as the map.

    Returns markup rather than rendering, so callers can place it inline in a
    sentence.
    """
    fill = theme.TIER_COLOURS.get(tier, theme.NEUTRAL[200])
    ink = theme.TIER_INK.get(tier, theme.INK)
    return f'<span class="badge" style="background:{fill};color:{ink}">{tier}</span>'


def download_figure(figure: Figure, basename: str) -> None:
    """Offer a matplotlib figure as a PNG and a PDF.

    Deliberately the **static** figure rather than the interactive one. Plotly's
    own image export needs a headless browser installed alongside the app, while
    matplotlib renders both formats in-process — and the static charts are already
    the report's figures, so the export and the report cannot drift apart.

    PDF as well as PNG because a PDF is vector: it survives being enlarged in a
    document, and a raster screenshot does not.
    """
    columns = st.columns(len(EXPORT_FORMATS), gap="small")
    for column, (extension, mime) in zip(columns, EXPORT_FORMATS.items(), strict=True):
        buffer = io.BytesIO()
        figure.savefig(buffer, format=extension, bbox_inches="tight", dpi=200)
        with column:
            st.download_button(
                extension.upper(),
                data=buffer.getvalue(),
                file_name=f"{basename}.{extension}",
                mime=mime,
                use_container_width=True,
                key=f"download_{basename}_{extension}",
            )


@contextmanager
def loading(message: str) -> Iterator[None]:
    """A spinner with a sentence, for the two panels that compute rather than read.

    Named rather than a bare spinner: "Loading" tells a reader nothing, while
    "Projecting six periods ahead" tells them what is slow and why.
    """
    with st.spinner(message):
        yield


def empty_state(message: str, hint: str = "") -> None:
    """An honest placeholder.

    Used wherever data may legitimately be absent. A blank area reads as a broken
    page; a sentence reads as a fact, and a hint tells the reader what would fix
    it.
    """
    body = message + (f"<br><br>{hint}" if hint else "")
    st.markdown(f'<div class="empty">{body}</div>', unsafe_allow_html=True)


def legend(bands: Sequence[tuple[str, str]], caption: str = "") -> None:
    """A legend with explicit numeric breaks, one swatch per band.

    Laid out horizontally and allowed to wrap. A vertical list of six swatches
    costs well over a hundred pixels, which on a fixed-height panel came straight
    out of the map above it and clipped the bottom row of states.
    """
    swatches = "".join(
        f'<div style="display:flex;align-items:center;gap:{theme.SPACE["xs"]}px">'
        f'<span style="width:{theme.SPACE["md"]}px;height:{theme.SPACE["md"]}px;'
        f"background:{theme.swatch_background(colour)};"
        f"border:{theme.BORDER_WIDTH}px solid {theme.BORDER};"
        f'border-radius:{theme.RADIUS}px;flex:none"></span>'
        f'<span style="font-size:{theme.TYPE["small"]}px;color:{theme.INK_MUTED};'
        f'white-space:nowrap">{label}</span></div>'
        for colour, label in bands
    )
    st.markdown(
        f'<div style="display:flex;flex-wrap:wrap;gap:{theme.SPACE["md"]}px;'
        f'margin-top:{theme.SPACE["sm"]}px">{swatches}</div>',
        unsafe_allow_html=True,
    )
    if caption:
        st.markdown(f'<div class="panel-note">{caption}</div>', unsafe_allow_html=True)


def evidence_list(items: Sequence[tuple[str, str]]) -> None:
    """Label and value pairs, for the numbers behind a recommendation."""
    rows = "".join(f"<div><b>{label}</b> {value}</div>" for label, value in items)
    st.markdown(f'<div class="evidence">{rows}</div>', unsafe_allow_html=True)


def note(text: str) -> None:
    """A small muted caption."""
    st.markdown(f'<div class="panel-note">{text}</div>', unsafe_allow_html=True)


def spacer(size: str = "md") -> None:
    """Vertical space from the 4px scale."""
    st.markdown(
        f'<div style="height:{theme.SPACE[size]}px"></div>', unsafe_allow_html=True
    )
