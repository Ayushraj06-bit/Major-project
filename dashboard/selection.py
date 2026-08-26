"""The selection store. One source of truth for what the page is showing.

Every panel renders from the same :class:`Selection`. No view keeps a copy, and
no view has a control of its own that changes what another view is describing.
That is the whole point: a page where the map shows October and the panel beside
it shows December is not a dashboard, it is two dashboards.

The store holds:

* ``state`` — the area in focus.
* ``period`` — the target period every panel describes.
* ``horizon`` — how far past the last observation to project, or ``0`` for the
  fitted series alone.
* ``show_uncertainty`` — whether interval bands are drawn.
* ``compare`` — additional states overlaid on the forecast chart.

Streamlit reruns the entire script on every interaction, so "state management"
here means two things and no more: read the widgets once into a frozen object,
and make the expensive work depend on that object rather than on the render. The
projection is cached on ``(state, period, horizon)`` in :mod:`dashboard.data`, so
dragging the slider and releasing it recomputes once, and re-rendering for a
scroll or a resize recomputes nothing.

Streamlit's ``select_slider`` commits on release rather than during the drag,
which is where a debounce would otherwise be needed. The play control is the one
place that changes the period without a user gesture, and it advances one period
per rerun for the same reason: a loop that raced ahead of the cache would queue
projections nobody asked for.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import streamlit as st

#: Projection lengths offered in the rail, as label to number of periods.
#:
#: Named rather than a free number box. "How many periods ahead" invites a reader
#: to type 24, and the honest answer to 24 is a refusal, so the control should not
#: ask the question. The cap itself lives in ``config.yaml``.
HORIZON_MODES: dict[str, int] = {
    "Project 3 periods": 3,
    "Project 6 periods": 6,
    "Fitted only": 0,
}

#: The mode the page opens on. **Projecting, deliberately.**
#:
#: This used to default to "Fitted only", which meant the forward projection --
#: the thing the page exists to show -- rendered nothing until a reader noticed
#: an unremarkable radio button and changed it. An empty chart is indistinguishable
#: from a broken one, and the feature read as missing rather than as off.
DEFAULT_HORIZON_MODE = "Project 3 periods"

#: Session keys. Collected so nothing else in the package spells one out.
KEY_STATE = "sel_state"
KEY_PERIOD = "sel_period"
KEY_HORIZON = "sel_horizon"
KEY_UNCERTAINTY = "sel_uncertainty"
KEY_COMPARE = "sel_compare"
KEY_PLAYING = "sel_playing"

#: Where an out-of-band selection is parked until the rail can adopt it.
#:
#: Streamlit refuses to let a widget-backed key be written **after** that widget
#: has been instantiated in the same run. The map and the watchlist sit below the
#: rail, so by the time a click is handled the Area selectbox already exists and
#: writing ``KEY_STATE`` directly raises. The click parks its choice here instead,
#: and :func:`apply_pending` adopts it at the top of the next run, before any
#: widget is drawn.
KEY_PENDING_STATE = "sel_pending_state"

#: Seconds between frames while the period is playing.
#:
#: Streamlit has no frame clock, so the loop is a rerun per step and this is the
#: pause before each one. Without it the loop spins as fast as the server can
#: render, which is neither watchable nor kind to the CPU. Just under a second
#: reads as deliberate stepping rather than as a flicker.
PLAY_INTERVAL_SECONDS = 0.9


@dataclass(frozen=True)
class Selection:
    """What the page is showing. Frozen, so no panel can quietly change it.

    Attributes:
        state: The area in focus.
        period: The target period every panel describes.
        horizon: Periods to project past the last observation; ``0`` for none.
        show_uncertainty: Whether prediction intervals are drawn.
        compare: Other states overlaid on the forecast chart.
        playing: Whether the period is advancing on its own.
    """

    state: str
    period: pd.Timestamp
    horizon: int = 0
    show_uncertainty: bool = True
    compare: tuple[str, ...] = field(default_factory=tuple)
    playing: bool = False

    @property
    def projecting(self) -> bool:
        """Whether a forward projection was asked for."""
        return self.horizon > 0

    @property
    def overlay(self) -> tuple[str, ...]:
        """States on the comparison chart, focus first and no duplicates."""
        others = tuple(name for name in self.compare if name != self.state)
        return (self.state, *others)

    def key(self) -> tuple[str, str, int]:
        """A hashable identity for caching work that depends on this selection.

        Deliberately excludes ``show_uncertainty`` and ``compare``: neither
        changes what is computed, only what is drawn. Including them would throw
        away cached work every time someone ticked a checkbox.

        Individual caches may key on less than this --
        :func:`dashboard.data.forecast_curve` ignores the period, because a
        projection always starts from the last observation -- but nothing should
        key on more.
        """
        return (self.state, self.period.isoformat(), self.horizon)


def comparison_options(states: list[str], focus: str) -> list[str]:
    """States offered for comparison: everything except the one already in focus.

    Also prunes the focus state out of the stored choice. Streamlit raises when a
    multiselect's stored value holds something its options no longer contain, so
    selecting a state that was already ticked for comparison would otherwise
    break the rail rather than simply move the state.
    """
    stored = list(st.session_state.get(KEY_COMPARE) or ())
    if focus in stored:
        st.session_state[KEY_COMPARE] = [name for name in stored if name != focus]
    return [name for name in states if name != focus]


def select_state(name: str) -> None:
    """Point the whole page at a state, from anywhere on the page.

    Parks the choice in :data:`KEY_PENDING_STATE` rather than writing
    :data:`KEY_STATE`, because the callers -- the map's click handler and the
    watchlist's buttons -- run after the rail's selectbox has been instantiated,
    and Streamlit raises on a widget key written that late. The caller reruns; the
    rail adopts it via :func:`apply_pending` before drawing anything.
    """
    st.session_state[KEY_PENDING_STATE] = name


def apply_pending(states: list[str]) -> None:
    """Adopt a state parked by :func:`select_state`. Call before drawing widgets.

    Silently drops a parked state the artifact no longer contains, which is the
    right behaviour after a rebuild: the page renders on a valid state rather than
    failing on a stale one.
    """
    pending = st.session_state.pop(KEY_PENDING_STATE, None)
    if pending is not None and pending in states:
        st.session_state[KEY_STATE] = pending


def advance_period(periods: list[pd.Timestamp]) -> None:
    """Step the period forward one, wrapping at the end.

    One step per rerun rather than an animation loop. Streamlit has no frame
    clock, and a loop that advanced faster than the projection cache could fill
    would spend its time computing curves that were already stale.

    **Must be called before the period widget is instantiated.** It writes a
    widget-backed key, and Streamlit raises if that happens after the widget
    exists -- which is exactly what pressing Play used to do.
    """
    if not periods:
        return
    current = st.session_state.get(KEY_PERIOD)
    labels = [period_label(period) for period in periods]
    position = labels.index(current) if current in labels else len(labels) - 1
    st.session_state[KEY_PERIOD] = labels[(position + 1) % len(labels)]


def toggle_play() -> None:
    """Start or stop the period advancing."""
    st.session_state[KEY_PLAYING] = not st.session_state.get(KEY_PLAYING, False)


def period_label(period: pd.Timestamp) -> str:
    """The human label for a period. One spelling, used by every control."""
    return pd.Timestamp(period).strftime("%b %Y")


def read(states: list[str], periods: list[pd.Timestamp]) -> Selection:
    """Assemble the current selection from session state.

    Reads rather than renders: the widgets are drawn by
    :func:`dashboard.app.rail`, which writes into these keys. Keeping the two
    apart means a test can build a :class:`Selection` without a running
    Streamlit script.
    """
    state = st.session_state.get(KEY_STATE) or (states[0] if states else "")
    if state not in states and states:
        state = states[0]

    labels = {period_label(period): period for period in periods}
    label = st.session_state.get(KEY_PERIOD)
    period = labels.get(label) if label else None
    if period is None:
        period = periods[-1] if periods else pd.Timestamp.today()

    horizon = HORIZON_MODES.get(
        st.session_state.get(KEY_HORIZON, DEFAULT_HORIZON_MODE), 0
    )
    compare = tuple(st.session_state.get(KEY_COMPARE) or ())

    return Selection(
        state=str(state),
        period=pd.Timestamp(period),
        horizon=int(horizon),
        show_uncertainty=bool(st.session_state.get(KEY_UNCERTAINTY, True)),
        compare=compare,
        playing=bool(st.session_state.get(KEY_PLAYING, False)),
    )
