# QA audit — dengue outbreak prediction pipeline

**Date:** 2026-08-26 · **Commit state:** post Phase 10b, 390 tests green before this audit
**Scope:** static analysis, correctness properties, edge cases, determinism, UI automation
**Method:** every claim below is backed by a reproduction command. Where I could not
reproduce something, it is downgraded and said so.

---

## Summary

| Severity | Count | |
|---|---|---|
| Critical | 0 | |
| Major | 2 | QA-1 silently wrong numbers · QA-2 unhandled crash path |
| Minor | 4 | QA-3…QA-6 |
| Observation | 2 | not defects; worth a decision |

**No leakage was found**, and the two properties most likely to be quietly broken —
interval calibration and simulation identity — both hold with margin. The two major
findings are a config option that half the codebase ignores, and an exception that
escapes the handler written to catch it.

---

## MAJOR

### QA-1 — `data.target_transform` is ignored by six call sites

**What's wrong.** `config.yaml` exposes `data.target_transform`, validated against
`{"log1p", "none"}` (`src/config.py:114`). Four modules honour it. **Six call sites
apply `np.expm1` unconditionally**, so setting the option to `none` exponentiates an
already-linear rate and reports the result as a case rate. No crash, no warning —
just wrong thresholds, wrong tiers, wrong recommendations and a wrong observed
history on the dashboard.

There is also no single inverse-transform function. The same three-line decision is
written **eight times**, four of them guarded and four not, so a change has to be
made in eight places and the next person will miss the same four.

| Site | Guarded? |
|---|---|
| `src/evaluate.py:214`, `:336` | yes |
| `src/features.py:769` | yes |
| `src/production.py:110` | yes |
| `src/recommend.py:232` | **no** |
| `src/recommend.py:265` | **no** |
| `src/recommend.py:342` | **no** |
| `src/recommend.py:378` (`_rate`) | **no** |
| `scripts/build_dashboard_data.py:40` | **no** |
| `dashboard/views.py:450` | **no** |

**Reproduce.**
```bash
pytest tests/test_qa_audit.py::test_every_inverse_transform_honours_the_config
```
Passes a log-scale value of `2.0` through `src.recommend._rate`. Expected `2.0`
under `target_transform="none"`; **got `6.389`** (`expm1(2.0)`).

**Fix.** Add one function — `src/features.py::inverse_target_transform(values, cfg)`
— and route all eight sites through it. It is the natural companion to the existing
`target_level`, which already owns the forward direction.

**Why major, not critical.** The shipped `config.yaml` uses `log1p`, so current
numbers are correct. The defect is latent: it fires the moment anyone exercises a
documented, validated option.

---

### QA-2 — `ProductionError` escapes the handler written to catch it

**What's wrong.** `dashboard/data.py:275` guards the forward-projection path with:

```python
except (SimulationError, FileNotFoundError, KeyError):
    # A state with too little history, or no frozen model. Both are states the
    # interface has to render rather than crash on.
```

`ProductionError` is a bare `RuntimeError` subclass and is **not** in that tuple.
It is raised by `ProductionModel._require_matching_columns` whenever the panel does
not produce the features the frozen model was fitted on — which is precisely the
"artifact and model disagree" case the comment claims to handle. The result is an
unhandled traceback in the forecast panel instead of a labelled empty state.

`FeatureError` has the same shape and is likewise uncaught, though I could not reach
it through the UI path (see *Not verified*).

**Reproduce.**
```bash
python - <<'PY'
from src.config import load_config
from src.panel import load_panel
from src.preprocess import preprocess
from src.production import load_production
from src.simulate import forecast_horizon
import pandas as pd
cfg = load_config(); model = load_production()
panel = load_panel(cfg, synthetic=True)
clean = preprocess(panel, cfg).panel[list(panel.columns)]
forecast_horizon(clean.drop(columns=["search_interest"]), "Delhi",
                 pd.Timestamp("2024-06-01"), model, cfg)
PY
# ProductionError: this panel does not produce the features the production model was fitted on
```

**How a user reaches it.** Re-freeze the production model with a different
`--experiment`, or edit `config.yaml`, without re-running
`build_dashboard_data.py`. The stored dashboard panel then no longer matches the
frozen model. `RUNNING.md` and brain.md both document re-freezing as a routine
single-step operation, so this window is a normal operational state, not an exotic one.

**Fix.** Add `ProductionError` and `FeatureError` to the tuple, and have the empty
state say *why* it is empty rather than rendering a blank chart — "this state's
features no longer match the frozen model; re-run `build_dashboard_data.py`" is
actionable, a blank panel is not.

---

## MINOR

### QA-3 — Circular import between `src.panel` and `src.synthetic`

`src.synthetic` imports `complete_index` from `src.panel`; `src.panel.load_panel`
imports `synthetic_panel` back. It works only because the second import is deferred
inside the function body. Static analysis flags it, and it is a trap for anyone who
later hoists that import to module scope.

**Reproduce.** The import-graph walker in *Track 1* reports
`['src.panel', 'src.synthetic', 'src.panel']`.

**Fix.** Move `complete_index` into a module that neither depends on — it is a pure
function of the config and has no business living beside the fusion logic.

---

### QA-4 — `FeatureError` misdiagnoses a spatial-lag cascade

Spatial lags are `NaN` for a state whose neighbours are all outside the study, which
correctly drops that state. But the drop **cascades**: starving one state's cases
also drops each of its neighbours. When that empties the panel, the error says:

```
no complete samples: sequence_length=3, horizon=1, max lag=12.
Each state needs enough consecutive observed periods to fill one window...
```

Every noun in that message is wrong for this cause. The user will tune
`sequence_length` indefinitely while the actual fix is `include_spatial: false` or a
wider study area.

**Reproduce.** Three states (Kerala, Odisha, Tamil Nadu), full history, then set
Kerala's `cases` to `NaN` after the first period. Odisha is already dropped
(no in-study neighbours), Tamil Nadu follows Kerala, and the panel empties.

Verified as *intended and recorded* behaviour in the non-cascading case:
`spec.dropped_states == ('Odisha',)`, and disabling `include_spatial` restores all
three states. So the mechanism is right; only the diagnosis is misleading.

**Fix.** When `dropped_states` is non-empty, name them and the reason in the error.

---

### QA-5 — Optional data-acquisition dependencies are undeclared

`cdsapi`, `xarray` and `trendspy` are imported (deferred, with actionable error
messages) by `src/sources/climate.py` and `src/sources/trends.py`, but appear in
neither `requirements.txt` nor `pyproject.toml`. Anyone fetching real data has to
discover them from a runtime error.

**Fix.** A `data = ["cdsapi", "xarray", "netCDF4", "trendspy"]` extra in
`pyproject.toml`, referenced from `RUNNING.md`.

---

### QA-6 — Two harness defects that manufactured twelve phantom findings

Recorded because both produced convincing-looking application failures, and because
either would have put fabricated defects into this report.

**Run 1 — 8 failures, all spurious.** One environmental hiccup
(`ERR_NETWORK_IO_SUSPENDED`) landed on a module-scoped error list that was never
drained, so every test after it inherited the error. Fixed: `_assert_healthy` drains
the list and filters sandbox-only messages.

**Run 2 — 4 failures, three spurious.** The server was launched with
`stdout=subprocess.PIPE` and nothing ever read it. Streamlit and TensorFlow log
steadily, the OS buffer filled, and the child blocked — mid-suite the page started
reporting *"Is Streamlit still running?"*. Three tests failed as if the app had
crashed. Fixed: `stdout=subprocess.DEVNULL`.

The fourth was a genuine test bug: the map-click test chose the first forecastable
tile without excluding the tile already selected, and clicking the current selection
is a deliberate no-op (`views._apply_map_click` guards on `name != current`). Now
picks a different state.

**Run 3 — 13 passed, 0 failed.**

The lesson worth carrying: a failing test is not evidence of a defect until the
harness is ruled out. Twelve of the sixteen failures across these runs were mine.

---

## OBSERVATIONS — not defects, but decisions worth making

### OBS-1 — The mandated model is the third-best model

On the synthetic panel: GBM **0.0225**, Ridge **0.0225**, LSTM **0.0265**,
seasonal-naive 0.0282. The gate compares the LSTM against *seasonal-naive only* and
prints `PASS`, and `select_configuration` restricts to `production.model` by design
(documented in its docstring, and the LSTM is fixed by the project brief).

Nothing here is hidden. But `PASS: LSTM beats seasonal naive` is easy to read as
"the LSTM is the best model", which on this data it is not. The write-up should
state the full table, not the gate alone.

### OBS-2 — README makes no reproducible numeric claims

Asked to confirm the README's figures reproduce: **it contains none.** It is a
specification. Its one numeric passage (line 221, "predicted 88 cases … 90th
percentile of 62") is an illustrative mock-up of the output format.

The project's real numeric claims live in `brain.md`. The headline one was checked
and **reproduces exactly**: LSTM MAE `0.0265` vs seasonal-naive `0.0282`, from a
clean run into a scratch results directory.

---

## VERIFIED CLEAN — with the evidence

These were attacked and held. Listed because "we tested for it" is worth as much as
a finding.

**Leakage** — `tests/test_qa_audit.py`
- Folds cut on **dates**, not positions; every state appears on both sides of every
  fold (a positional split would produce a state holdout that scores well).
- Embargo strictly exceeds the horizon in every fold.
- **Mutation-tested**: multiplying all rainfall after time *t* by 10 leaves every
  feature at or before *t* bit-identical. This catches an off-by-one in a shift,
  which inspecting column names would not.

**Feature purity** — called twice on identical input returns identical output;
does not mutate its input; interleaving two different panels four times shows no
cross-call state.

**Simulation** — `tests/test_qa_model.py`
- A 0% scenario reproduces the baseline at `atol=0`, **exactly**.
- Changing raw rainfall moves *all* of its derived columns — current value, lags,
  rolling mean and spatial term — verified per-column on the built tensor.
- The OOD flag discriminates: +2000% fires it and clamps >50% of cells; +1% does not.

**Interval calibration** — 83.3% observed against 80% nominal over 1728 rows.
Checked for vacuity: shrinking the interval 10× drops coverage to 12.4%, so the
test would fail if the layer broke.

**Recursive projection** — visible interval width (upper minus lower, not the
internal half-width) strictly increases at every step; reliability decays
monotonically; the cap truncates and says so; no non-finite or negative output.

**Artifact integrity** — network input shape `(None, 12, 29)` matches
`spec.timesteps=12` and `spec.n_features=29`; `scaler_mean`/`scaler_scale` both have
29 entries; no stored scale is zero (`StandardScaled.fit` substitutes 1.0 for a
constant column — load-bearing, now pinned by a test).

**State normalisation** — Orissa→Odisha, Uttaranchal→Uttarakhand,
Pondicherry→Puducherry, "NCT of Delhi"→Delhi, whitespace and case variants all
collapse; normalisation is idempotent; an unknown name raises `UnknownStateError`
rather than being dropped.

**Determinism** — two clean runs into separate scratch directories:
- baselines: `metrics.json` **byte-identical** (≈11 s each)
- **LSTM: byte-identical** (≈45 s each) — Keras training is reproducible here,
  which many projects of this shape are not

**UI, automated** — `tests/test_dashboard_e2e.py`, 13 tests, all passing, driven in
Chromium under a **dark** OS preference (the widget-theme defect was invisible in
light mode, so testing light alone would have missed it):
- Every control exercised: area selectbox, period slider, play/pause, map click,
  projection radio, uncertainty checkbox, compare multiselect, scenario run, export.
- **Cross-panel sync asserted**, not eyeballed: after changing the area, a panel is
  titled with the new state; after moving the period, map and watchlist name the
  *same* month; after a map click, the rail's selectbox matches the panel.
- Widget-label contrast measured in-browser against the page background: must clear
  4.5:1. This is the check that would have caught the white-on-white defect.
- Adversarial: five rapid tile clicks in 750 ms, and switching state while a
  projection is still computing. Both settle correctly with no stale panel.

**Hardcoding and duplication** — no state names, paths, thresholds or magic numbers
outside config in `src/`, `dashboard/` or `scripts/`. `.shift(`/`.rolling(` appear
**only** in `src/features.py`. No second scaler implementation. The one duplication
found is QA-1's inverse transform.

---

## NOT VERIFIED — honest gaps

1. **Fresh-venv install.** `pip install -e .` was verified in the *existing*
   environment. I did not create a clean virtualenv and install from
   `requirements.txt` alone, so an undeclared transitive dependency could still be
   hiding. QA-5 was found by static analysis, not by a clean install.

2. **`FeatureError` reaching the UI.** QA-2's sibling. I confirmed `FeatureError`
   is not caught, but could not construct a panel that raises it through
   `forecast_horizon` with the current 12-state configuration — starving one state,
   and starving all-but-one, both returned clean curves. It is a latent gap in the
   handler, not a demonstrated crash.

3. **Real data.** Everything is synthetic (`data/raw/` is empty). Interval coverage,
   the LSTM gate and every threshold describe a sine-wave generator. Coverage on
   real dengue data could differ substantially, and the recursive `sqrt(depth)`
   widening is an *approximation* that has never been calibrated against anything.

4. **Full-pipeline double run.** I ran `run_baselines` and `run_lstm` twice each.
   I did **not** run `run_ablations` → `freeze_production` → `run_shap` twice; SHAP
   in particular is stochastic (KernelExplainer sampling) and its determinism is
   unverified.

5. **Manual UI judgements.** Automation confirmed the controls *work*; it cannot
   judge appearance. Whether the interval band *visibly* widens, whether the
   direct/recursive boundary is distinguishable at a glance, and whether the loading
   spinner reads as progress rather than as a hang — these need a human.
   `MANUAL_QA_CHECKLIST.md` covers them; I have not marked them off.

6. **Cross-browser.** Chromium only. No Firefox or WebKit.

7. **Accessibility beyond contrast.** I measured widget-label contrast
   programmatically. Screen-reader behaviour, focus order and full keyboard
   traversal are unaudited.

---

## Deliverables

| File | What |
|---|---|
| `tests/test_qa_audit.py` | Leakage, purity, transforms, normalisation, edge cases. Fast. |
| `tests/test_qa_model.py` | Calibration, simulation, recursion, artifact integrity. Needs the frozen model; skips with a reason if absent. Marked `slow`. |
| `tests/test_dashboard_e2e.py` | Launches Streamlit, drives Chromium, exercises every control and asserts cross-panel sync. **13 passed.** Marked `e2e`; skips if Playwright is absent. |
| `MANUAL_QA_CHECKLIST.md` | Click-through list for the judgements automation cannot make. |

```bash
pytest tests/test_qa_audit.py                  # fast
pytest tests/test_qa_model.py -m slow          # needs results/runs/production
pytest tests/test_dashboard_e2e.py -m e2e      # needs playwright + chromium
```
