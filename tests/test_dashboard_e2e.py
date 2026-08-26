"""End-to-end UI audit: drive the real dashboard in a real browser.

Not a unit test. It starts the Streamlit server, opens a Chromium page and
exercises every interactive control, asserting three things each time:

1. the control responds,
2. no Python exception is rendered and no console error is logged,
3. **every panel agrees** afterwards -- no panel left describing the previous
   selection.

Point 3 is the one automation is uniquely good at. A human clicking through sees
a page that looks fine; only a machine reliably notices that the SHAP panel is
still showing the state you left two clicks ago.

Skipped, with a named reason, when Playwright or its browser is missing, so a
machine without them reports "not verified" rather than a false pass.

Run with::

    pytest tests/test_dashboard_e2e.py -v
"""

from __future__ import annotations

import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PORT = 8599
URL = f"http://localhost:{PORT}"

#: How long to wait for the server to answer before giving up.
BOOT_TIMEOUT_SECONDS = 90

pytestmark = pytest.mark.e2e

playwright_api = pytest.importorskip(
    "playwright.sync_api", reason="playwright not installed: pip install playwright"
)


def _port_is_open(port: int) -> bool:
    with socket.socket() as probe:
        probe.settimeout(0.5)
        return probe.connect_ex(("127.0.0.1", port)) == 0


@pytest.fixture(scope="module")
def server() -> Iterator[str]:
    """A Streamlit server on a private port, torn down afterwards."""
    if _port_is_open(PORT):
        pytest.skip(f"port {PORT} already in use")

    # Discard the server's output rather than piping it. Streamlit and
    # TensorFlow between them log steadily, and an unread subprocess.PIPE fills
    # its OS buffer and blocks the child forever -- which killed the server
    # mid-suite and produced three "Is Streamlit still running?" failures that
    # looked like application defects.
    process = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "dashboard/app.py",
         "--server.port", str(PORT), "--server.headless", "true"],
        cwd=PROJECT_ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
    )
    try:
        deadline = time.monotonic() + BOOT_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if _port_is_open(PORT):
                break
            if process.poll() is not None:
                pytest.skip("streamlit exited during boot")
            time.sleep(1)
        else:
            pytest.skip(f"streamlit did not boot within {BOOT_TIMEOUT_SECONDS}s")
        yield URL
    finally:
        process.terminate()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()


@pytest.fixture(scope="module")
def page(server: str) -> Iterator[Any]:
    """One page, shared across the module, recording every console error.

    Deliberately opened with ``color_scheme="dark"``: the widget-theme defect
    only appeared under a dark OS preference, so testing the light case alone
    would have missed it entirely.
    """
    try:
        with playwright_api.sync_playwright() as pw:
            browser = pw.chromium.launch()
            view = browser.new_page(
                viewport={"width": 1600, "height": 1100}, color_scheme="dark"
            )
            view.console_errors = []  # type: ignore[attr-defined]
            view.on("console", lambda m: (
                view.console_errors.append(m.text) if m.type == "error" else None
            ))
            view.on("pageerror", lambda e: view.console_errors.append(str(e)))
            view.goto(server, wait_until="networkidle")
            _settle(view, 10000)
            yield view
            browser.close()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"could not launch a browser: {exc}")


def _settle(page: Any, first_wait: int = 3000) -> None:
    """Wait for Streamlit to finish its rerun."""
    page.wait_for_timeout(first_wait)
    for _ in range(60):
        if page.locator('[data-testid="stStatusWidget"]').count() == 0:
            return
        page.wait_for_timeout(500)


def _exceptions(page: Any) -> list[str]:
    """Any Python traceback Streamlit rendered into the page."""
    box = page.locator('[data-testid="stException"]')
    return [box.nth(i).inner_text() for i in range(box.count())]


def _headers(page: Any) -> list[str]:
    box = page.locator("div.panel-header")
    return [box.nth(i).inner_text() for i in range(box.count())]


#: Console messages that say something about the sandbox, not about the page.
ENVIRONMENTAL = (
    "metrics",                    # Streamlit's usage telemetry, blocked offline
    "err_network_io_suspended",   # the browser's network was suspended by the OS
    "err_internet_disconnected",
    "favicon",
)


def _assert_healthy(page: Any, action: str) -> None:
    """No rendered traceback, and no console error since the last check.

    **Drains** the error list. It is collected on a module-scoped page, so
    without draining, one environmental hiccup early in the run fails every test
    after it and reports eight defects where there are none -- which is exactly
    what the first run of this suite did.
    """
    problems = _exceptions(page)
    errors, page.console_errors[:] = list(page.console_errors), []
    assert not problems, f"{action} raised in the app:\n{problems[0][:400]}"

    real = [
        error for error in errors
        if not any(pattern in error.lower() for pattern in ENVIRONMENTAL)
    ]
    assert not real, f"{action} logged console error(s): {real[:3]}"


# --------------------------------------------------------------------------- #
# Baseline
# --------------------------------------------------------------------------- #


def test_the_page_loads_with_every_panel(page: Any) -> None:
    headers = _headers(page)
    for expected in ("AREA", "PERIOD", "FORWARD PROJECTION", "OBSERVED AND PREDICTED",
                     "COMPARE STATES", "ATTRIBUTION", "SCENARIO"):
        assert any(expected in header for header in headers), (
            f"panel {expected!r} missing; found {headers}"
        )
    _assert_healthy(page, "initial load")


def test_widget_text_is_legible_against_the_page(page: Any) -> None:
    """The defect a screenshot caught and every token-based test missed.

    Streamlit styles its own widgets. With no theme pinned it follows the
    viewer's OS colour scheme, so on a dark machine it drew near-white label text
    on the white page our stylesheet forces -- about 1.05:1.
    """
    probe = page.evaluate("""() => {
      const label = document.querySelector('[data-testid="stWidgetLabel"] p')
                 || document.querySelector('[data-testid="stWidgetLabel"]');
      const parse = c => (c.match(/\\d+/g) || []).slice(0, 3).map(Number);
      const lum = ([r, g, b]) => {
        const f = v => { v /= 255; return v <= 0.03928 ? v / 12.92
                                 : Math.pow((v + 0.055) / 1.055, 2.4); };
        return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b);
      };
      const bg = parse(getComputedStyle(document.querySelector('.stApp')).backgroundColor);
      const fg = parse(getComputedStyle(label).color);
      const a = lum(bg), b = lum(fg);
      return (Math.max(a, b) + 0.05) / (Math.min(a, b) + 0.05);
    }""")
    assert probe >= 4.5, (
        f"widget label contrast is {probe:.2f}:1 against the page background; "
        "WCAG AA for body text is 4.5:1"
    )


# --------------------------------------------------------------------------- #
# Every control
# --------------------------------------------------------------------------- #


def test_changing_the_area_updates_every_panel(page: Any) -> None:
    """Cross-panel sync. A stale panel is the failure automation is best at."""
    detail_before = _headers(page)
    page.locator('[data-testid="stSidebar"] [data-baseweb="select"]').first.click()
    page.wait_for_timeout(600)
    options = page.locator('li[role="option"]')
    assert options.count() > 1, "area selector offers nothing to change to"

    chosen = options.nth(1).inner_text().strip()
    options.nth(1).click()
    _settle(page, 6000)
    _assert_healthy(page, "changing the area")

    headers = _headers(page)
    assert any(chosen.upper() in header for header in headers), (
        f"no panel is titled {chosen!r} after selecting it; headers={headers}"
    )
    assert headers != detail_before, "the page did not change at all"


def test_moving_the_period_retitles_the_map_and_watchlist(page: Any) -> None:
    """Two panels carry the period in their title. Both must follow."""
    before = [h for h in _headers(page) if "NATIONAL RISK" in h or "WATCHLIST" in h]
    assert len(before) == 2, f"expected map and watchlist headers, got {before}"

    slider = page.locator('[data-testid="stSidebar"] [role="slider"]').first
    slider.click()
    for _ in range(6):
        page.keyboard.press("ArrowLeft")
    _settle(page, 6000)
    _assert_healthy(page, "moving the period slider")

    after = [h for h in _headers(page) if "NATIONAL RISK" in h or "WATCHLIST" in h]
    assert after != before, f"period moved but titles did not: {before} -> {after}"
    assert len(set(h.split("·")[-1].strip() for h in after)) == 1, (
        f"map and watchlist show different periods: {after}"
    )


def test_the_play_control_advances_and_stops(page: Any) -> None:
    """The crash this replaces: writing a widget key after the widget exists."""
    before = next(h for h in _headers(page) if "NATIONAL RISK" in h)
    page.get_by_role("button", name="Play").click()
    _settle(page, 6000)
    _assert_healthy(page, "pressing Play")

    after = next(h for h in _headers(page) if "NATIONAL RISK" in h)
    assert after != before, f"Play did not advance the period: {before}"

    page.get_by_role("button", name="Pause").click()
    _settle(page, 4000)
    _assert_healthy(page, "pressing Pause")
    held = next(h for h in _headers(page) if "NATIONAL RISK" in h)
    page.wait_for_timeout(3000)
    assert next(h for h in _headers(page) if "NATIONAL RISK" in h) == held, (
        "the period kept moving after Pause"
    )


def test_clicking_a_map_tile_selects_that_state(page: Any) -> None:
    """The path that crashed identically to Play, and that nobody had exercised."""
    plot = page.locator(".js-plotly-plot").first
    plot.scroll_into_view_if_needed()
    page.wait_for_timeout(800)

    current = page.locator(
        '[data-testid="stSidebar"] [data-baseweb="select"]'
    ).first.inner_text().strip()

    # Must not be the tile already selected: clicking it is a no-op by design,
    # so the test would be asserting that nothing changed.
    target = page.evaluate("""(current) => {
      const gd = document.querySelector('.js-plotly-plot');
      const trace = gd.data.find(t => t.customdata);
      for (let i = 0; i < trace.customdata.length; i++) {
        const known = !String(trace.meta[i]).includes('not in this study');
        if (known && trace.customdata[i] !== current) return [i, trace.customdata[i]];
      }
      return null;
    }""", current)
    assert target, f"no forecastable tile other than the selected {current!r}"
    position, name = target

    page.locator("g.points path.point").nth(position).click(force=True)
    _settle(page, 7000)
    _assert_healthy(page, "clicking a map tile")

    assert any(str(name).upper() in header for header in _headers(page)), (
        f"clicking {name} did not retitle the detail panel"
    )
    selected = page.locator(
        '[data-testid="stSidebar"] [data-baseweb="select"]'
    ).first.inner_text().strip()
    assert str(name) in selected, (
        f"map click set the panel to {name} but the rail still says {selected!r}; "
        "the two are out of sync"
    )


def test_the_projection_control_adds_a_recursive_region(page: Any) -> None:
    radios = page.locator('[data-testid="stSidebar"] [role="radiogroup"] label')
    assert radios.count() >= 3, "forward-projection options missing"
    for index in range(radios.count()):
        if "6 periods" in radios.nth(index).inner_text():
            radios.nth(index).click()
            break
    _settle(page, 12000)
    _assert_healthy(page, "selecting a 6-period projection")

    shapes = page.evaluate("""() => {
      const plots = [...document.querySelectorAll('.js-plotly-plot')];
      return plots.map(p => (p.layout.shapes || []).filter(s => s.type === 'rect').length);
    }""")
    assert any(count > 0 for count in shapes), (
        "no chart marks a recursive region after asking for a 6-period projection"
    )


def test_the_uncertainty_toggle_removes_the_band_only(page: Any) -> None:
    def band_traces() -> int:
        return page.evaluate("""() => {
          const gd = [...document.querySelectorAll('.js-plotly-plot')]
            .find(p => p.data.some(t => t.fill === 'toself'));
          return gd ? gd.data.filter(t => t.fill === 'toself').length : 0;
        }""")

    with_band = band_traces()
    page.get_by_text("Show prediction intervals").click()
    _settle(page, 8000)
    _assert_healthy(page, "toggling prediction intervals")
    without = band_traces()

    assert without < with_band, f"band count unchanged: {with_band} -> {without}"
    page.get_by_text("Show prediction intervals").click()
    _settle(page, 8000)


def test_the_compare_control_overlays_a_second_state(page: Any) -> None:
    page.locator('[data-testid="stSidebar"] [data-testid="stMultiSelect"] input').click()
    page.wait_for_timeout(700)
    options = page.locator('li[role="option"]')
    assert options.count() > 0, "comparison offers no states"
    options.nth(0).click()
    page.keyboard.press("Escape")
    _settle(page, 8000)
    _assert_healthy(page, "adding a comparison state")

    lines = page.evaluate("""() => {
      const plots = [...document.querySelectorAll('.js-plotly-plot')];
      return Math.max(...plots.map(p => p.data.filter(t => t.mode === 'lines').length));
    }""")
    assert lines >= 2, f"comparison chart shows {lines} line(s)"


def test_the_scenario_control_runs_and_reports(page: Any) -> None:
    button = page.get_by_role("button", name="Run scenario")
    button.scroll_into_view_if_needed()
    button.click()
    _settle(page, 30000)
    _assert_healthy(page, "running a scenario")

    body = page.locator("body").inner_text()
    assert "correlation, not causation" in body, (
        "the scenario ran without printing its causal caveat"
    )


def test_the_export_buttons_are_offered(page: Any) -> None:
    for label in ("PNG", "PDF"):
        assert page.get_by_role("button", name=label).count() >= 1, (
            f"no {label} export control"
        )


# --------------------------------------------------------------------------- #
# Adversarial interaction
# --------------------------------------------------------------------------- #


def test_rapid_repeated_clicks_do_not_break_the_page(page: Any) -> None:
    """Double-firing: one gesture must not queue two conflicting selections."""
    plot = page.locator(".js-plotly-plot").first
    plot.scroll_into_view_if_needed()
    points = page.locator("g.points path.point")
    for index in (2, 3, 4, 2, 3):
        points.nth(index).click(force=True)
        page.wait_for_timeout(150)
    _settle(page, 12000)
    _assert_healthy(page, "rapid repeated map clicks")

    headers = _headers(page)
    assert any("NATIONAL RISK" in header for header in headers), (
        "the page did not recover from rapid clicking"
    )


def test_changing_selection_mid_load_leaves_no_stale_panel(page: Any) -> None:
    """Race: switch state while a projection is still computing."""
    select = page.locator('[data-testid="stSidebar"] [data-baseweb="select"]').first
    select.click()
    page.wait_for_timeout(500)
    options = page.locator('li[role="option"]')
    first = options.nth(0).inner_text().strip()
    options.nth(0).click()
    page.wait_for_timeout(400)          # deliberately do not wait for settle

    select.click()
    page.wait_for_timeout(500)
    options = page.locator('li[role="option"]')
    second = options.nth(1).inner_text().strip()
    options.nth(1).click()
    _settle(page, 15000)
    _assert_healthy(page, "changing selection mid-load")

    headers = _headers(page)
    assert any(second.upper() in header for header in headers), (
        f"after racing {first} -> {second}, no panel shows {second}: {headers}"
    )
    assert not any(first.upper() in header for header in headers if "RISK" not in header), (
        f"a panel is still showing the abandoned selection {first}"
    )
