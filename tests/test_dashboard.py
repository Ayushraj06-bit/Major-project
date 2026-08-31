"""Dashboard: the design rules, and the failure modes an interface must survive."""

from __future__ import annotations

import ast
import dataclasses
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dashboard import charts, components, data, plots, selection, theme, views
from dashboard.geo import TILE_POSITIONS
from src.sources.registry import CANONICAL_NAMES

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DASHBOARD = PROJECT_ROOT / "dashboard"

#: Everything except the token module, which is the one place values may be written.
OTHER_MODULES = sorted(
    path for path in DASHBOARD.glob("*.py") if path.name not in {"theme.py", "__init__.py"}
)

HEX_COLOUR = re.compile(r"#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})\b")
PIXEL_VALUE = re.compile(r"\b\d+px\b")


# --------------------------------------------------------------------------- #
# Review gate: no colour or pixel value outside theme.py
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("path", OTHER_MODULES, ids=lambda p: p.name)
def test_no_colour_literal_outside_theme(path: Path) -> None:
    """The gate. One drifting grey is how an interface stops looking designed."""
    found = HEX_COLOUR.findall(_code_only(path))
    assert not found, f"{path.name} writes colour(s) {found}; put them in theme.py"


@pytest.mark.parametrize("path", OTHER_MODULES, ids=lambda p: p.name)
def test_no_pixel_literal_outside_theme(path: Path) -> None:
    """Sizes come from the 4px scale, not from wherever felt right at the time."""
    found = PIXEL_VALUE.findall(_code_only(path))
    assert not found, f"{path.name} writes size(s) {found}; use theme.SPACE"


def test_theme_is_the_only_module_defining_the_scale() -> None:
    """A second spacing scale would defeat having one."""
    for path in OTHER_MODULES:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        assigned = {
            target.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        assert "SPACE" not in assigned, f"{path.name} defines its own spacing scale"
        assert "NEUTRAL" not in assigned


# --------------------------------------------------------------------------- #
# Review gate: hierarchy survives with the accent covered
# --------------------------------------------------------------------------- #


def _luminance(hex_colour: str) -> float:
    """Relative luminance, for the greyscale check."""
    value = hex_colour.lstrip("#")
    channels = [int(value[i : i + 2], 16) / 255 for i in (0, 2, 4)]
    linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def test_hierarchy_is_carried_by_size_not_colour() -> None:
    """Cover the accent and the structure must still read.

    Hierarchy comes from the type scale and the spacing scale. If the only thing
    separating a title from body text were colour, the layout would be doing too
    little work.
    """
    sizes = sorted(theme.TYPE.values())
    assert len(set(sizes)) == 3, "three sizes, no more"
    # Each step must be a visible jump, not a nudge.
    for smaller, larger in zip(sizes, sizes[1:], strict=False):
        assert larger / smaller >= 1.2, f"{smaller}px to {larger}px is not a step"

    assert len(set(theme.WEIGHT.values())) == 2, "two weights, no more"


def test_text_greys_stay_distinguishable_in_greyscale() -> None:
    """The three ink levels must separate without any hue to help them."""
    inks = [theme.INK, theme.INK_MUTED, theme.INK_FAINT]
    luminances = [_luminance(colour) for colour in inks]
    assert luminances == sorted(luminances), "ink levels must get lighter in order"
    for darker, lighter in zip(luminances, luminances[1:], strict=False):
        assert lighter - darker > 0.05, "ink levels are too close to tell apart"


def test_body_text_meets_contrast_on_the_page_background() -> None:
    """WCAG AA for body text is 4.5:1. Muted ink is still text."""
    background = _luminance(theme.SURFACE)
    # Every ink level, including the faintest. Tick labels and captions are text.
    for ink in (theme.INK, theme.INK_MUTED, theme.INK_FAINT):
        ratio = (max(_luminance(ink), background) + 0.05) / (
            min(_luminance(ink), background) + 0.05
        )
        assert ratio >= 4.5, f"{ink} on {theme.SURFACE} is {ratio:.1f}:1"


def test_the_accent_is_used_once_and_only_for_interaction() -> None:
    """One accent. A second would stop the first meaning anything."""
    source = (DASHBOARD / "theme.py").read_text(encoding="utf-8")
    accents = {theme.ACCENT, theme.ACCENT_STRONG, theme.ACCENT_WASH}
    assert len(accents) == 3, "accent, its active state, and its wash"

    # The accent must not appear in the risk ramp, or selection and severity
    # would be told apart only by position.
    assert theme.ACCENT not in theme.RISK_RAMP

    # No decorative gradients: a smooth fade is ornament, and ornament competes
    # with the one colour that is supposed to mean something. A *repeating*
    # gradient is exempt because it is not a fade -- it is how CSS draws the
    # no-data hatch, which exists to remove an ambiguity rather than to decorate.
    decorative = re.findall(r"(?<!repeating-)(?:linear|radial)-gradient", source)
    assert not decorative, f"theme.py uses decorative gradient(s) {decorative}"


# --------------------------------------------------------------------------- #
# The risk ramp
# --------------------------------------------------------------------------- #


def test_risk_ramp_is_sequential_in_luminance() -> None:
    """A ramp whose lightness does not order is a rainbow with extra steps.

    Monotone luminance is what makes it survive greyscale printing and red-green
    colour deficiency.
    """
    luminances = [_luminance(colour) for colour in theme.RISK_RAMP]
    assert luminances == sorted(luminances, reverse=True), (
        "risk ramp must darken monotonically as risk rises"
    )


def test_risk_colour_respects_explicit_breaks() -> None:
    breaks = [1.0, 2.0, 3.0, 4.0]
    assert theme.risk_colour(0.5, breaks) == theme.RISK_RAMP[0]
    assert theme.risk_colour(3.5, breaks) == theme.RISK_RAMP[3]
    assert theme.risk_colour(99.0, breaks) == theme.RISK_RAMP[-1]


def test_legend_states_every_band_numerically() -> None:
    """A reader must be able to say what a colour means, not estimate it."""
    bands = charts.legend_bands([1.0, 2.0, 3.0, 4.0])
    # One per risk band, plus the no-data swatch.
    assert len(bands) == len(theme.RISK_RAMP) + 1
    assert all(any(character.isdigit() for character in label) for _, label in bands[:-1])
    assert "and above" in bands[-2][1]
    assert bands[-1] == (theme.NO_DATA, "not in this study")


# --------------------------------------------------------------------------- #
# Review gate: survives a state with missing data
# --------------------------------------------------------------------------- #


def test_forecast_chart_renders_a_placeholder_when_empty() -> None:
    """The gate: an honest placeholder, never a crash or a blank void."""
    empty = pd.DataFrame(columns=["date", "actual"])
    figure = charts.forecast_chart(empty, pd.DataFrame(columns=["date", "predicted"]))
    assert figure is not None
    assert _text_of(figure), "an empty chart must say why it is empty"


def test_attribution_chart_renders_a_placeholder_when_empty() -> None:
    figure = charts.attribution_chart([], [])
    assert "No cached attributions" in _text_of(figure)


def test_tile_map_renders_a_placeholder_when_empty() -> None:
    figure = charts.tile_map({}, TILE_POSITIONS, [1.0, 2.0, 3.0, 4.0])
    assert "No forecasts" in _text_of(figure)


def test_charts_survive_a_series_that_is_all_missing() -> None:
    """A state present in the panel but with no observations at all."""
    history = pd.DataFrame(
        {"date": pd.date_range("2020-01-01", periods=6, freq="MS"), "actual": [np.nan] * 6}
    )
    forecast = pd.DataFrame(
        {
            "date": pd.date_range("2020-07-01", periods=3, freq="MS"),
            "predicted": [np.nan] * 3,
            "lower": [np.nan] * 3,
            "upper": [np.nan] * 3,
        }
    )
    assert charts.forecast_chart(history, forecast) is not None


def test_tile_map_ignores_states_it_has_no_position_for() -> None:
    """A state outside the grid must not take the map down with it."""
    figure = charts.tile_map(
        {"Kerala": 1.0, "Atlantis": 5.0}, TILE_POSITIONS, [1.0, 2.0, 3.0, 4.0]
    )
    assert figure is not None


def test_map_draws_every_state_not_only_those_with_forecasts() -> None:
    """States outside the study are shown as no-data rather than omitted.

    Omitting them left the map a fragment of the country, and with only twelve
    states occupying eight rows it became tall enough to push the selected state
    out of the panel entirely.
    """
    figure = charts.tile_map({"Kerala": 1.0}, TILE_POSITIONS, [1.0, 2.0, 3.0, 4.0])
    axes = figure.get_axes()[0]
    assert len(axes.patches) == len(TILE_POSITIONS)


def test_no_data_tiles_are_hatched_not_merely_a_different_grey() -> None:
    """'Not studied' must never read as 'low risk'.

    Colour alone cannot separate them: the lightest risk band is brighter than
    any usable grey, so a plain grey tile would sit between the risk bands rather
    than outside them. The hatch is what makes it unambiguous, including in
    greyscale.
    """
    figure = charts.tile_map({"Kerala": 1.0}, TILE_POSITIONS, [1.0, 2.0, 3.0, 4.0])
    patches = figure.get_axes()[0].patches

    hatched = [patch for patch in patches if patch.get_hatch()]
    assert len(hatched) == len(TILE_POSITIONS) - 1, "every state but Kerala is no-data"
    assert all(patch.get_hatch() == theme.NO_DATA_HATCH for patch in hatched)


# --------------------------------------------------------------------------- #
# Review gate: the dashboard computes nothing it should have read
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("path", OTHER_MODULES, ids=lambda p: p.name)
def test_dashboard_does_not_train_or_explain(path: Path) -> None:
    """The gate. Simulation is the single permitted live computation."""
    tree = ast.parse(path.read_text(encoding="utf-8"))

    forbidden_calls = {"fit", "fit_transform", "shap_values", "train_on_batch"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr not in forbidden_calls, (
                f"{path.name} calls .{node.func.attr}(); the dashboard reads results"
            )

    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    banned = {
        "build_features", "run_experiment", "train_production", "explain",
        "compute_thresholds", "recommend", "run_ablations", "PooledLSTM",
    }
    assert not (imported & banned), (
        f"{path.name} imports {sorted(imported & banned)}; that belongs in "
        "scripts/build_dashboard_data.py"
    )


#: The only two things the dashboard is allowed to compute at render time.
#:
#: Neither can be precomputed -- a what-if depends on inputs chosen on screen,
#: and a projection depends on a horizon chosen on screen. Everything else was
#: written by ``scripts/build_dashboard_data.py``. A third entry here should be
#: argued for, not added.
LIVE_PATHS = {"run_scenario", "forecast_curve"}


def test_the_live_paths_are_named_and_live_in_data() -> None:
    """Computation is permitted in two places, and must stay in those two."""
    source = (DASHBOARD / "data.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions = {
        node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }
    assert LIVE_PATHS.issubset(functions)

    for name in ("simulate", "forecast_horizon"):
        assert name in source, f"data.py no longer routes to {name}"


def test_no_view_computes_a_forecast_itself() -> None:
    """A view that called the model directly would bypass the cache.

    The whole point of routing through :mod:`dashboard.data` is that the work
    happens once per selection change rather than once per render, and Streamlit
    re-runs every view on every interaction.
    """
    tree = ast.parse((DASHBOARD / "views.py").read_text(encoding="utf-8"))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert "forecast_horizon" not in imported
    assert "load_production" not in imported


def test_the_projection_cache_ignores_display_only_settings() -> None:
    """Ticking a checkbox must not throw away a computed projection."""
    import inspect

    from dashboard.data import forecast_curve

    parameters = set(inspect.signature(forecast_curve).parameters)
    assert parameters == {"state", "horizon"}, (
        f"forecast_curve keys on {sorted(parameters)}; anything beyond state and "
        "horizon recomputes work that did not change"
    )


def test_scenario_view_shows_a_spinner() -> None:
    """It re-runs the feature pipeline, so it must not look frozen."""
    source = (DASHBOARD / "views.py").read_text(encoding="utf-8")
    assert "st.spinner" in source


# --------------------------------------------------------------------------- #
# Layout and structure
# --------------------------------------------------------------------------- #


def test_every_state_has_a_tile_position() -> None:
    """A state with no tile would silently vanish from the map."""
    assert set(TILE_POSITIONS) == set(CANONICAL_NAMES)


def test_no_two_states_share_a_tile() -> None:
    assert len(set(TILE_POSITIONS.values())) == len(TILE_POSITIONS)


def test_panel_heights_are_fixed_so_the_page_does_not_reflow() -> None:
    """Switching states must not make the page jump."""
    assert set(theme.PANEL_HEIGHT) >= {"map", "chart", "detail"}
    assert all(isinstance(value, int) for value in theme.PANEL_HEIGHT.values())


def test_spacing_scale_is_multiples_of_four() -> None:
    assert all(value % 4 == 0 for value in theme.SPACE.values())
    assert sorted(theme.SPACE.values()) == [4, 8, 16, 24, 32]


def test_charts_carry_no_chartjunk() -> None:
    """Two spines, no frame, no shadow."""
    rc = theme.mpl_rc()
    assert rc["axes.spines.top"] is False
    assert rc["axes.spines.right"] is False
    assert rc["legend.frameon"] is False


def test_stylesheet_interpolates_tokens_rather_than_hardcoding() -> None:
    """The CSS is generated, so changing a token changes the interface."""
    stylesheet = theme.css()
    assert theme.ACCENT in stylesheet
    assert f"{theme.SPACE['md']}px" in stylesheet
    assert f"{theme.TYPE['title']}px" in stylesheet


def _code_only(path: Path) -> str:
    """Module source with docstrings stripped.

    Prose describing the 4px scale is documentation, not a hardcoded value, and a
    check that cannot tell them apart trains people to stop reading it.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docstrings = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and id(node) in docstrings:
            node.value = ""
    return ast.unparse(tree)


def _text_of(figure) -> str:
    """All text drawn on a figure, for asserting on placeholders."""
    return " ".join(
        text.get_text() for axes in figure.get_axes() for text in axes.texts
    )


# --------------------------------------------------------------------------- #
# The selection store: one source of truth for what the page shows
# --------------------------------------------------------------------------- #


def _selection(**overrides: object) -> selection.Selection:
    """A selection with sensible defaults, for the store's own tests."""
    fields: dict[str, object] = {
        "state": "Kerala",
        "period": pd.Timestamp("2023-06-01"),
        "target_date": pd.Timestamp("2024-01-01"),
    }
    fields.update(overrides)
    return selection.Selection(**fields)  # type: ignore[arg-type]


def test_selection_is_frozen_so_no_panel_can_change_it() -> None:
    """The invariant the whole layout rests on.

    A panel that could write to the selection is a panel that can change what its
    neighbours are describing, which is exactly the bug the store exists to stop.
    """
    current = _selection()
    with pytest.raises(dataclasses.FrozenInstanceError):
        current.state = "Odisha"  # type: ignore[misc]


def test_overlay_puts_the_focus_first_and_never_twice() -> None:
    """The selected state is on the comparison chart whether or not it was ticked."""
    current = _selection(compare=("Odisha", "Kerala"))
    assert current.overlay == ("Kerala", "Odisha")


def test_cache_key_ignores_what_only_changes_the_drawing() -> None:
    """Uncertainty and comparison change the render, not the computation."""
    plain = _selection()
    decorated = _selection(show_uncertainty=False, compare=("Odisha",))
    assert plain.key() == decorated.key()


def test_cache_key_separates_what_does_change_the_computation() -> None:
    assert (
        _selection(target_date=pd.Timestamp("2024-01-01")).key()
        != _selection(target_date=pd.Timestamp("2024-04-01")).key()
    )
    assert _selection(state="Kerala").key() != _selection(state="Odisha").key()


def test_the_picker_offers_months_the_model_cannot_reach() -> None:
    """Deliberate. A refusal that explains itself teaches the model's limits.

    A greyed-out control that says nothing reads as a broken one, so the picker
    accepts the question and :func:`~src.simulate.classify_target` answers with a
    reason instead of a number.
    """
    assert selection.YEARS_BEYOND_REACH > 0
    assert len(selection.MONTHS) == 12


def test_advance_period_wraps_at_the_end(monkeypatch: pytest.MonkeyPatch) -> None:
    """The play control loops rather than stopping silently at the last period."""
    periods = [pd.Timestamp("2023-01-01"), pd.Timestamp("2023-02-01")]
    store: dict[str, object] = {selection.KEY_PERIOD: "Feb 2023"}
    monkeypatch.setattr(selection.st, "session_state", store)

    selection.advance_period(periods)
    assert store[selection.KEY_PERIOD] == "Jan 2023"


def test_read_falls_back_when_the_stored_state_is_gone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rebuilt artifact can drop a state. The page must render, not crash."""
    store: dict[str, object] = {selection.KEY_STATE: "Atlantis"}
    monkeypatch.setattr(selection.st, "session_state", store)

    current = selection.read(
        ["Kerala", "Odisha"], [pd.Timestamp("2023-01-01")], pd.Timestamp("2024-01-01")
    )
    assert current.state == "Kerala"
    assert current.period == pd.Timestamp("2023-01-01")
    assert current.target_date == pd.Timestamp("2024-01-01")


# --------------------------------------------------------------------------- #
# Interactive charts: what a static image could not say
# --------------------------------------------------------------------------- #


def _curve(recursive_from: int = 1, steps: int = 4) -> pd.DataFrame:
    """A projection frame with a direct head and a recursive tail."""
    dates = pd.date_range("2024-01-01", periods=steps, freq="MS")
    widths = [0.1 * (index + 1) for index in range(steps)]
    return pd.DataFrame(
        {
            "date": dates,
            "predicted": [1.0] * steps,
            "lower": [1.0 - width for width in widths],
            "upper": [1.0 + width for width in widths],
            "mode": ["direct"] * recursive_from
            + ["recursive"] * (steps - recursive_from),
        }
    )


def test_recursive_steps_are_drawn_differently_from_direct_ones() -> None:
    """The most important thing the chart says.

    A projection where the model reads its own output must not look like one
    built from measured inputs. Tested on the figure rather than trusted to a
    caption, because a reader glances at the shape before reading anything.
    """
    figure = plots.forecast_chart(
        history=pd.DataFrame(columns=["date", "actual"]),
        forecast=pd.DataFrame(columns=["date", "predicted"]),
        projection=_curve(),
    )
    dashed = [
        trace for trace in figure.data
        if getattr(trace, "line", None) is not None and trace.line.dash == "dot"
    ]
    assert dashed, "the forward projection is not visually distinguished"

    shaded = [shape for shape in figure.layout.shapes if shape.type == "rect"]
    assert shaded, "the recursive region is not marked on the axis"


def test_a_fully_direct_projection_is_not_shaded() -> None:
    """Shading must mean something. Marking a direct forecast would make it noise."""
    figure = plots.forecast_chart(
        history=pd.DataFrame(columns=["date", "actual"]),
        forecast=pd.DataFrame(columns=["date", "predicted"]),
        projection=_curve(recursive_from=4, steps=4),
    )
    assert not [shape for shape in figure.layout.shapes if shape.type == "rect"]


def test_hiding_uncertainty_removes_the_band_not_the_line() -> None:
    """The checkbox is a display setting, not a different forecast."""
    frame = pd.DataFrame(
        {
            "target_date": pd.date_range("2023-01-01", periods=3, freq="MS"),
            "predicted_cases_per_100k": [1.0, 2.0, 3.0],
            "lower_cases_per_100k": [0.5, 1.5, 2.5],
            "upper_cases_per_100k": [1.5, 2.5, 3.5],
        }
    )
    shown = views._forecast_frame(frame, show_uncertainty=True)
    hidden = views._forecast_frame(frame, show_uncertainty=False)

    assert {"lower", "upper"} <= set(shown.columns)
    assert not {"lower", "upper"} & set(hidden.columns)
    pd.testing.assert_series_equal(shown["predicted"], hidden["predicted"])


def test_every_interactive_chart_survives_being_given_nothing() -> None:
    """Empty is a normal state, and every panel has to render a sentence for it."""
    empty = pd.DataFrame(columns=["date", "predicted"])
    figures = [
        plots.forecast_chart(pd.DataFrame(columns=["date", "actual"]), empty),
        plots.comparison_chart({}),
        plots.attribution_chart([], []),
        plots.scenario_chart(empty, empty),
        plots.tile_map({}, TILE_POSITIONS, [1.0, 2.0, 3.0, 4.0]),
    ]
    for figure in figures:
        assert figure.layout.annotations, "an empty chart must say why it is empty"


def test_the_map_carries_every_state_and_reports_the_unstudied_ones() -> None:
    """A tile with no forecast must say so on hover, not show a bare number."""
    figure = plots.tile_map(
        {"Kerala": 1.0}, TILE_POSITIONS, [1.0, 2.0, 3.0, 4.0], selected="Kerala"
    )
    trace = _point_trace(figure)
    assert len(trace.customdata) == len(TILE_POSITIONS)
    assert any("not in this study" in str(value) for value in trace.meta)


def test_the_map_returns_a_state_name_when_a_tile_is_clicked() -> None:
    """Click-to-select needs the state name on the point, not its index."""
    figure = plots.tile_map({"Kerala": 1.0}, TILE_POSITIONS, [1.0, 2.0, 3.0, 4.0])
    assert set(_point_trace(figure).customdata) == set(TILE_POSITIONS)


def test_the_comparison_chart_colours_only_the_focus() -> None:
    """One accent, spent on the selection. The rest are context."""
    frame = pd.DataFrame(
        {"date": pd.date_range("2023-01-01", periods=3, freq="MS"),
         "predicted": [1.0, 2.0, 3.0]}
    )
    figure = plots.comparison_chart({"Kerala": frame, "Odisha": frame}, focus="Kerala")
    accented = [
        trace for trace in figure.data if trace.line.color == theme.ACCENT
    ]
    assert len(accented) == 1


def test_both_chart_libraries_read_the_same_tokens() -> None:
    """Two libraries, one visual language.

    The interactive charts and the export charts have to agree, or the page and
    the report look like two different projects.
    """
    layout = theme.plotly_template()["layout"]
    assert layout["paper_bgcolor"] == theme.SURFACE
    assert layout["font"]["color"] == theme.INK
    assert layout["colorway"][0] == theme.ACCENT
    assert layout["transition"]["duration"] == theme.TRANSITION_MS


def test_one_transition_duration_for_everything_that_moves() -> None:
    """Two speeds on one page read as a bug rather than as a style."""
    source = (DASHBOARD / "theme.py").read_text(encoding="utf-8")
    assert source.count("TRANSITION_MS = ") == 1
    assert source.count("EASING = ") == 1


# --------------------------------------------------------------------------- #
# The watchlist
# --------------------------------------------------------------------------- #


def _point_trace(figure: object) -> object:
    """The map's interactive trace: the one carrying a state name per point."""
    return next(
        trace for trace in figure.data  # type: ignore[attr-defined]
        if trace.customdata is not None
    )


def _dataset(forecasts: pd.DataFrame, thresholds: pd.DataFrame) -> data.DashboardData:
    """A dataset carrying only what the watchlist reads."""
    return data.DashboardData(
        forecasts=forecasts,
        recommendations=pd.DataFrame(),
        thresholds=thresholds,
        history=pd.DataFrame(),
        panel=pd.DataFrame(),
        meta={},
    )


def test_watchlist_ranks_by_exceedance_not_by_raw_rate() -> None:
    """Otherwise the list sorts by population density instead of by concern."""
    period = pd.Timestamp("2023-06-01")
    forecasts = pd.DataFrame(
        {
            "state": ["Kerala", "Odisha"],
            "target_date": [period, period],
            "origin_date": [period, period],
            "upper_cases_per_100k": [10.0, 3.0],
        }
    )
    thresholds = pd.DataFrame(
        {
            "state": ["Kerala", "Odisha"],
            "tier": ["HIGH", "HIGH"],
            "value_cases_per_100k": [9.0, 1.0],
        }
    )
    rows = data.watchlist(_dataset(forecasts, thresholds), period)

    # Odisha is lower in absolute terms and further past its own threshold.
    assert list(rows["state"]) == ["Odisha", "Kerala"]


def test_watchlist_is_empty_when_nothing_crosses() -> None:
    """An empty watchlist is a result, not a missing panel."""
    period = pd.Timestamp("2023-06-01")
    forecasts = pd.DataFrame(
        {
            "state": ["Kerala"], "target_date": [period], "origin_date": [period],
            "upper_cases_per_100k": [1.0],
        }
    )
    thresholds = pd.DataFrame(
        {"state": ["Kerala"], "tier": ["HIGH"], "value_cases_per_100k": [9.0]}
    )
    assert data.watchlist(_dataset(forecasts, thresholds), period).empty


def test_watchlist_skips_a_state_with_no_threshold() -> None:
    """No threshold means no judgement, which is not the same as no concern."""
    period = pd.Timestamp("2023-06-01")
    forecasts = pd.DataFrame(
        {
            "state": ["Kerala"], "target_date": [period], "origin_date": [period],
            "upper_cases_per_100k": [99.0],
        }
    )
    thresholds = pd.DataFrame(
        {"state": ["Odisha"], "tier": ["HIGH"], "value_cases_per_100k": [1.0]}
    )
    assert data.watchlist(_dataset(forecasts, thresholds), period).empty


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #


def test_export_uses_the_report_figures_not_a_screenshot() -> None:
    """What a reader downloads has to be what the write-up will contain."""
    source = (DASHBOARD / "views.py").read_text(encoding="utf-8")
    assert "charts.forecast_chart" in source


def test_export_offers_a_vector_format() -> None:
    """A PNG enlarged in a document is a blurry PNG."""
    assert "pdf" in components.EXPORT_FORMATS


def test_the_interactive_map_hatches_states_it_has_no_forecast_for() -> None:
    """The regression this guards is subtle and was found by looking, not testing.

    A plain grey tile sits *between* the risk bands in luminance -- the lightest
    band is brighter than any usable grey -- so "not studied" reads as "low risk".
    The static map solved this with a hatch, and the interactive one has to keep
    it or the port quietly reintroduces the bug.
    """
    figure = plots.tile_map({"Kerala": 1.0}, TILE_POSITIONS, [1.0, 2.0, 3.0, 4.0])
    hatch = [trace for trace in figure.data if trace.mode == "lines"]
    assert hatch, "states with no forecast are not distinguished by a pattern"

    # One segment per offset per unforecast state, plus a None gap after each.
    unforecast = len(TILE_POSITIONS) - 1
    assert len(hatch[0].x) == unforecast * len(plots._HATCH_OFFSETS) * 3


def test_the_legend_swatch_matches_how_the_map_draws_it() -> None:
    """A plain swatch beside a hatched tile puts the ambiguity straight back."""
    assert theme.swatch_background(theme.NO_DATA) != theme.NO_DATA
    assert theme.swatch_background(theme.RISK_RAMP[0]) == theme.RISK_RAMP[0]


def test_a_long_series_opens_on_a_recent_window() -> None:
    """Otherwise a six-month projection is a few pixels against twelve years."""
    dates = pd.date_range("2012-01-01", periods=144, freq="MS")
    history = pd.DataFrame({"date": dates, "actual": np.linspace(1, 2, len(dates))})
    figure = plots.forecast_chart(
        history, pd.DataFrame(columns=["date", "predicted"])
    )
    assert figure.layout.xaxis.range is not None


def test_every_time_series_opens_on_the_same_window() -> None:
    """Two charts side by side with different spans invite a false comparison."""
    dates = pd.date_range("2012-01-01", periods=144, freq="MS")
    frame = pd.DataFrame({"date": dates, "predicted": np.linspace(1, 2, len(dates))})
    history = pd.DataFrame({"date": dates, "actual": np.linspace(1, 2, len(dates))})

    spans = [
        plots.forecast_chart(history, frame).layout.xaxis.range,
        plots.comparison_chart({"Kerala": frame}).layout.xaxis.range,
        plots.scenario_chart(frame, frame).layout.xaxis.range,
    ]
    assert all(span is not None for span in spans)
    assert len(set(spans)) == 1, f"charts open on different spans: {spans}"


def test_a_short_series_is_shown_whole() -> None:
    """Zooming into a two-year series would hide context for no benefit."""
    dates = pd.date_range("2022-01-01", periods=12, freq="MS")
    history = pd.DataFrame({"date": dates, "actual": np.linspace(1, 2, len(dates))})
    figure = plots.forecast_chart(
        history, pd.DataFrame(columns=["date", "predicted"])
    )
    assert figure.layout.xaxis.range is None


def test_comparison_never_offers_the_state_already_in_focus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Comparing a state with itself is a control that does nothing."""
    monkeypatch.setattr(selection.st, "session_state", {})
    offered = selection.comparison_options(["Kerala", "Odisha"], "Kerala")
    assert offered == ["Odisha"]


def test_selecting_a_compared_state_moves_it_rather_than_breaking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Streamlit raises when a stored choice is no longer an option.

    So focusing a state that was ticked for comparison has to prune it, or the
    rail crashes instead of simply moving the state into focus.
    """
    store: dict[str, object] = {selection.KEY_COMPARE: ["Kerala", "Odisha"]}
    monkeypatch.setattr(selection.st, "session_state", store)

    offered = selection.comparison_options(["Kerala", "Odisha"], "Kerala")
    assert "Kerala" not in offered
    assert store[selection.KEY_COMPARE] == ["Odisha"]


def test_the_projection_line_joins_the_fitted_series() -> None:
    """A one-period gap between two traces reads as missing data."""
    fitted = pd.DataFrame(
        {"date": pd.date_range("2023-10-01", periods=3, freq="MS"),
         "predicted": [1.0, 2.0, 3.0]}
    )
    projection = _curve()
    figure = plots.forecast_chart(
        pd.DataFrame(columns=["date", "actual"]), fitted, projection=projection
    )
    dotted = next(
        trace for trace in figure.data
        if getattr(trace, "line", None) is not None and trace.line.dash == "dot"
    )
    assert pd.Timestamp(dotted.x[0]) == fitted["date"].iloc[-1]
    assert dotted.y[0] == fitted["predicted"].iloc[-1]


# --------------------------------------------------------------------------- #
# Cost per rerun: Streamlit re-runs the whole script on every interaction
# --------------------------------------------------------------------------- #

#: Everything expensive enough that repeating it per interaction is a defect.
CACHED_ENTRY_POINTS = {"load", "load_attributions", "forecast_curve", "production"}


def test_the_expensive_entry_points_are_cached() -> None:
    """A dashboard that reloads a Keras model to redraw a checkbox feels broken.

    Streamlit re-runs the entire script on every widget interaction, so anything
    that reads a file or builds a model has to be cached or it is paid for on
    each one. Two of these once claimed caching in their docstrings while having
    none, which is how the cost went unnoticed.
    """
    tree = ast.parse((DASHBOARD / "data.py").read_text(encoding="utf-8"))
    decorated = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        for decorator in node.decorator_list
        if "cache" in ast.unparse(decorator)
    }
    assert CACHED_ENTRY_POINTS.issubset(decorated), (
        f"uncached: {sorted(CACHED_ENTRY_POINTS - decorated)}"
    )


def test_the_model_is_shared_rather_than_reloaded_per_caller() -> None:
    """Three call sites loading their own copy is three Keras models in memory."""
    tree = ast.parse((DASHBOARD / "data.py").read_text(encoding="utf-8"))
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "load_production"
    ]
    assert len(calls) == 1, (
        f"load_production is called {len(calls)} times; route every caller "
        "through production() so the model is built once"
    )


def test_no_view_spells_a_session_key() -> None:
    """The keys live in one module so a rename cannot half-apply."""
    for path in OTHER_MODULES:
        if path.name == "selection.py":
            continue
        source = _code_only(path)
        found = re.findall(r'"sel_[a-z]+"', source)
        assert not found, f"{path.name} spells session key(s) {found}"


def test_chart_heights_are_derived_from_tokens_not_arithmetic() -> None:
    """A bare offset beside a panel height is how two columns stop lining up."""
    source = _code_only(DASHBOARD / "views.py")
    found = re.findall(r'PANEL_HEIGHT\[[^\]]+\]\s*[-+]\s*\d+', source)
    assert not found, f"views.py adjusts a panel height by hand: {found}"


# --------------------------------------------------------------------------- #
# Streamlit's own widget theme
# --------------------------------------------------------------------------- #


def test_streamlit_widget_theme_is_pinned_to_the_tokens() -> None:
    """Widgets must not follow the viewer's OS colour scheme.

    Streamlit styles its own widgets, and with no theme configured it follows
    `prefers-color-scheme`. On a machine set to dark that drew near-white label
    text on the white page our stylesheet pins -- about 1.05:1, which is not poor
    contrast but invisible text -- and near-black input boxes beside it.
    """
    config = (PROJECT_ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8")
    settings = dict(
        line.split(" = ", 1) for line in config.splitlines() if " = " in line
    )
    expected = theme.streamlit_config()

    assert settings.keys() >= expected.keys(), "config.toml is missing theme settings"
    for key, value in expected.items():
        assert settings[key].strip('"') == value, (
            f"config.toml {key} has drifted from dashboard/theme.py"
        )


def test_the_widget_accent_is_the_project_accent() -> None:
    """Streamlit's default primary is red. One accent means one accent."""
    assert theme.streamlit_config()["primaryColor"] == theme.ACCENT


def test_the_widget_theme_does_not_track_the_viewer_os() -> None:
    assert theme.streamlit_config()["base"] == "light"


# --------------------------------------------------------------------------- #
# Writing session state at a legal time
# --------------------------------------------------------------------------- #


def test_out_of_band_selection_is_parked_not_written_directly() -> None:
    """Streamlit forbids writing a widget key after that widget exists.

    The map and the watchlist sit below the rail, so their click handlers run
    after the Area selectbox has been instantiated. Writing `sel_state` there
    raised `StreamlitAPIException` -- the same crash pressing Play produced.
    """
    store: dict[str, object] = {selection.KEY_STATE: "Delhi"}
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(selection.st, "session_state", store)
        selection.select_state("Kerala")

        assert store[selection.KEY_STATE] == "Delhi", (
            "select_state wrote the widget key directly; Streamlit raises on that"
        )
        assert store[selection.KEY_PENDING_STATE] == "Kerala"


def test_a_parked_selection_is_adopted_before_the_widgets_draw() -> None:
    store: dict[str, object] = {
        selection.KEY_STATE: "Delhi", selection.KEY_PENDING_STATE: "Kerala"
    }
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(selection.st, "session_state", store)
        selection.apply_pending(["Delhi", "Kerala"])

    assert store[selection.KEY_STATE] == "Kerala"
    assert selection.KEY_PENDING_STATE not in store


def test_a_parked_state_the_artifact_dropped_is_ignored() -> None:
    """A rebuilt artifact can lose a state. Render, do not fail."""
    store: dict[str, object] = {
        selection.KEY_STATE: "Delhi", selection.KEY_PENDING_STATE: "Atlantis"
    }
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(selection.st, "session_state", store)
        selection.apply_pending(["Delhi", "Kerala"])

    assert store[selection.KEY_STATE] == "Delhi"


def test_play_advances_the_period_before_the_slider_is_built() -> None:
    """The crash this guards: `advance_period` writes a widget-backed key.

    Called from the tail of `main` it ran after the slider existed and raised. It
    must be called from `rail`, above every widget.
    """
    source = (DASHBOARD / "app.py").read_text(encoding="utf-8")
    rail_at = source.index("def rail(")
    advance_at = source.index("selection.advance_period(")
    slider_at = source.index("st.select_slider(")

    assert rail_at < advance_at < slider_at, (
        "advance_period must run inside rail() and before the period slider"
    )


def test_the_play_loop_is_paced() -> None:
    """An unpaced rerun loop spins as fast as the server can render."""
    assert selection.PLAY_INTERVAL_SECONDS > 0
    assert "time.sleep(selection.PLAY_INTERVAL_SECONDS)" in (
        (DASHBOARD / "app.py").read_text(encoding="utf-8")
    )
