# brain.md — Working Memory

**Project:** Multi-Source Spatio-Temporal Dengue Outbreak Prediction using Explainable Deep Learning
**Created:** 2026-08-21
**Source of truth for the spec:** [README.md](README.md)

---

## 0. How this file is used

This is the persistent reference for the project. Operating contract:

1. **Read this file first** at the start of any work session, before touching code.
2. **README.md is the frozen spec.** brain.md is the living state: what is decided, what is open, what has changed, what is done.
3. **Every change gets logged** in §11 Change Log — code written, decisions made, spec deviations, improvements.
4. **When brain.md and README.md disagree, brain.md wins** — it records supersessions the README does not know about (see §3).
5. Open questions live in §5 with IDs (`Q-nn`). When one is answered, move it to §4 as a decision (`D-nn`) and log it.

---

## 1. Project identity — FIXED, do not renegotiate

| Field | Value |
|-------|-------|
| Title | *Multi-Source Spatio-Temporal Dengue Outbreak Prediction using Explainable Deep Learning* |
| Domain | Dengue outbreak forecasting, Indian states |
| Workflow | Predict → Explain → Simulate → Recommend → Visualize |
| Primary model | LSTM (required, not negotiable — README §5.3) |
| Contribution framing | Adapting an established framework (Chen & Moraga 2025, Brazil) to **India**, and extending it **from forecasting to decision support**. NOT architectural novelty. |

**Four questions the system answers:** What will happen (LSTM) · Why (SHAP) · What if (simulation) · What to do (decision support).

**Novelty claims that are defensible** (README §15 Positioning):
1. First application to India / monsoon-driven seasonality
2. Scenario simulation — absent from the reference literature
3. Decision support — absent from the reference literature
4. Public-awareness signals — reference work uses climate + spatial only

**Research questions:** RQ1 multi-source > single-source? (ablation) · RQ2 which lags/sources matter? (SHAP + lag ablation) · RQ3 can XAI explain a deep temporal model on epi data? · RQ4 sensitivity of risk to environmental change? (simulation)

---

## 2. Current status

| Item | State |
|------|-------|
| Phase | **Phases 0–10b code complete.** Our own data is still absent; the pipeline runs end to end only on a synthetic panel. Alka's real weekly UP data now exists but is a different granularity. |
| Code written | Everything through Phase 6: scaffold, sources, registry, fusion, quality, preprocessing, features, splits, harness, baselines, pooled LSTM, ablation runner, conformal intervals. |
| Repo scaffold | Created and verified — 390 tests passing, ruff and mypy clean. |
| Data located | **None.** No raw file has been obtained. Every source raises an actionable error naming the file it needs and where to get it. |
| Environment | Not set up |

**Nothing downstream can be designed until Phase 1 closes.** README §4 is explicit: data granularity determines sequence length, lag windows, dataset size, model capacity, and whether the dashboard is a district heatmap or a state choropleth.

---

## 3. Supersessions — where §15 overrides the earlier README

Read line by line, §15 (Planned Improvements) silently replaces decisions made in §5–§11. These are the *current* rules:

| # | Earlier README says | §15 says — **THIS WINS** | Ref |
|---|---------------------|--------------------------|-----|
| S-1 | Model granularity open: shared-with-state-feature *or* per-state (§5.2) | **Pooled multi-state model with state embedding.** Per-state LSTM is dead — ~60 monthly points per state is unusable. | §15.2 |
| S-2 | Prediction intervals via **MC dropout** (§9) | **Conformal prediction** — rolling-window residual quantiles. Distribution-free, cheaper, adapts as transmission changes. | §15.7 |
| S-3 | Static features: concat after recurrent layers *or* predict per-100k (§5.1) | **Predict cases per 100,000.** Also makes states comparable. | §15.2 |
| S-4 | `pytrends` for Google Trends (§11) | **pytrends is archived (April 2025) and breaks on first call.** Use `trendspy`, manual Trends CSV export, or Wikipedia Pageviews API. | §15.11 |
| S-5 | SHAP via DeepExplainer, fallback GradientExplainer (§7) | **DeepExplainer does not support TF 2.x recurrent layers.** Wrap the LSTM: flatten input → reshape inside predict fn → `KernelExplainer`. Version-proof, slow. | §15.12 |
| S-6 | Metrics: MAE, RMSE, R² (§6) | **Add CRPS** — scores interval calibration, not just point accuracy. | §15.8 |
| S-7 | Flat spatial mean of gridded climate → state (§4) | **Population-weighted aggregation.** A flat mean over a large state is dominated by empty area, not where people live. | §15.10 |
| S-8 | SHAP is the explanation layer (§7) | SHAP is **also a pipeline component** — use mean-|SHAP| to select top ~5 features. | §15.1 |

---

## 4. Locked decisions

| ID | Decision | Source |
|----|----------|--------|
| D-01 | LSTM is the primary model. Random Forest dropped from the core pipeline (optional baseline only). | §5.3 |
| D-02 | Target transform: **log(x + 1)** on the rate. Combined with S-3 → target is `log(cases_per_100k + 1)`. Stabilises variance; stops one outbreak dominating the loss. | §5.2 + §15.2/15.4 |
| D-03 | Validation: **time-based splitting only**. Random shuffle = leakage = invalid results. | §5.4 |
| D-04 | **Rolling-origin cross-validation**, report mean ± std across folds. A single split on ~150 points is a coin flip. | §5.4 |
| D-05 | **Scale inside each fold** — fit scaler on the train portion only. Scaling the full series up front is the classic leakage bug examiners look for. | §5.4 |
| D-06 | Baselines are mandatory: persistence, seasonal-naive, linear/GBM on the same lagged features. **If the LSTM cannot beat seasonal naive, that is a month-2 finding, not a month-6 one.** | §6 |
| D-07 | Ablation grid is over **data configurations, not algorithms**: A climate-only / B +cases / C full multi-source; D no-lags / E lags / F lags+spatial. | §6 |
| D-08 | Scenario simulator must run `modify raw series → re-run full feature engineering → predict`. Never edit the final feature vector directly. | §8 |
| D-09 | Simulator inputs clamped to plausible range (historical min/max or ±2σ), with a warning when the user exits it. | §8 |
| D-10 | Recommendation thresholds must be **data-derived** (historical quantiles, EWMA, or Farrington) — never a hand-written if-else table. | §9 |
| D-11 | Alerts trigger on the **interval upper bound**, not the mean. Appropriate for public-health preparedness. | §9 + §15.7 |
| D-12 | Public-awareness signal = search/attention data (Trends or Wikipedia Pageviews). **Not** social-media scraping. | §4 |
| D-13 | Cyclic seasonality features `sin(2πt/T)`, `cos(2πt/T)` are included. | §15.3 |
| D-14 | Network is **small**: 32–64 LSTM units, dropout, early stopping. The reference paper's 1,000 units would memorise an Indian dataset this size. | §15.5 |
| D-15 | Spatial lag features are **effectively Core, not optional** — see C-1 below. | §14 |
| D-16 | Prediction intervals are **effectively Core, not optional** — see C-2 below. | §9 |
| D-17 | SHAP must be prototyped in **week 1** against a dummy LSTM, in parallel with everything else. Longest-standing known failure point in this stack. | §7, §15.13 |
| D-18 | Report negative ablation results honestly. In Brazil spatial features *hurt* in sparsely-connected northern states; expect the same in India's northeast. Knowing when a method fails is a finding. | §15.9 |

### Training config (starting point, §5.3)
Max epochs 100 · early stopping patience 10 · batch size 32 · Adam · MSE loss (on the log-rate target).

---

## 5. Open questions — must be closed

### Contradictions found in the README (I resolved these; confirm)

| ID | Conflict | My resolution |
|----|----------|---------------|
| C-1 | §3 lists spatial lag features as *Recommended* (row 12), but §14 says the fixed title claims "spatio-temporal" and a temporal-only model is **overclaiming**. | **Promote to Core.** The title is fixed, so the spatial component is not optional. Logged as D-15. |
| C-2 | §3 lists prediction intervals as *Recommended* (row 13), but §9's target output format requires them ("80% interval upper bound 140") and D-11 triggers alerts on the upper bound. | **Promote to Core.** Logged as D-16. |
| C-3 | §5.2 says forecast horizon **4 periods ahead**; §15.6 says forecast at **1 and 3 months**. | Unresolvable until granularity is known (Q-01). If data is weekly → 4 weeks ≈ 1 month, both agree. If monthly → report **1-month and 3-month** horizons. |
| C-4 | §15.1 says select top-5 features by SHAP **per state**; §15.2 mandates a single **pooled** model with shared input dimensions. Per-state feature sets cannot feed one pooled model. | Use pooled model with a **global** top-k feature set. Report per-state SHAP rankings as a *finding* ("which drivers matter where") rather than as per-state input surgery. |
| C-5 | §5.2 sets target `log(cases+1)`; §15.2 sets target `cases per 100,000`. | Combine: **`log(cases_per_100k + 1)`**. Logged as D-02. |
| C-6 | §4 says Trends is weekly/state-level; §4's critical note expects case data to be monthly. | Aggregate weekly Trends → monthly (mean and max) once granularity is fixed. |

### Needs the user / guide to answer

| ID | Question | Why it blocks |
|----|----------|---------------|
| Q-01 | **Is Indian dengue case data available weekly, or only monthly/annually? At state or district level?** | Determines sequence length, lag windows, dataset size, LSTM capacity, dashboard granularity, and C-3. **The single largest project risk.** |
| Q-02 | Submission deadline / how many weeks available? | Determines whether items 11–14 in §3 (Recommended/Optional) get built. |
| Q-03 | Solo or team? How many people? | Determines parallelisation of the 11-phase roadmap. |
| Q-04 | Has the guide mandated a tech stack, or is §11 free to follow? | §11 is currently a *suggestion*. |
| Q-05 | Which states / how many, and what year range? | Sets dataset size. Pooled model needs enough states to justify the embedding. |
| Q-06 | Streamlit or Dash for the dashboard? | §11 leaves it open. Streamlit is faster to build; Dash gives finer layout control. |
| Q-07 | Is the recommendation layer's action catalogue (source reduction, bed pre-positioning, awareness messaging) to be cited from an official source (NVBDCP/WHO guidelines) or authored by us? | §9 is flagged "design not finalised" and is **the part most likely to be probed in review** because it came from guide feedback. Citing an official source is far more defensible. |
| Q-08 | Are MAE/RMSE/R² reported on the **log scale**, **back-transformed to cases per 100k**, or both? | `exp(y)-1` back-transformation is biased by Jensen's inequality, but log-scale MAE is uninterpretable to a public-health reader. Shapes A-3d and every table in the report. Recommend: report both, note the bias. |
| Q-10 | Lag features and sequence windowing encode the same history twice. Keep both (612 flat inputs / 1728 samples), or set `sequence_length: 1`, or `include_lags: false`? | Measured under the shipped config, not estimated. Ratio 2.8 invites memorisation on a dataset this size; the alternatives give 36.5 and 15.4. Both are config-only changes. Decide before Phase 5 — after the LSTM loses to seasonal-naive is too late to tell whether capacity or signal was the problem. |
| Q-09 | For early stopping (patience 10, D-14), does the LSTM carve its validation set from the **tail of the train fold** internally, or does `.fit()` take an explicit `val` argument? | An explicit argument breaks the uniform A-2 protocol; an internal time-ordered carve keeps it. Recommend the internal carve, with the fraction from config. Must be time-ordered either way — a random carve reintroduces D-03. |

---

## 6. Hard constraints & known footguns

Violating any of these breaks the research validity, not just the code.

1. **No random shuffling. Ever.** Time-based splits only. (D-03)
2. **No scaler fit on the full series.** Fit per-fold, on train only. (D-05)
3. **Simulator must re-run feature engineering.** If rainfall changes, every `rainfall_lag_*` and rolling mean must move with it. Editing one cell of the final feature vector produces an input the model never saw in training, and its response is meaningless. (D-08)
4. **Correlation, not causation.** "Risk rises when rainfall rises" describes the *model's learned behaviour*, not dengue biology. Stating this in the report converts a limitation into a discussion point — a strength. (§8)
5. **Don't extrapolate.** Beyond the training distribution is exactly where neural networks produce confident nonsense. Clamp and warn. (D-09)
6. **Small dataset, small network.** ~100–150 time points per state. Pooling across states is what makes the LSTM viable at all (~2,000 rows). (D-14, S-1)
7. **SHAP on sequence models is slow and fragile.** Prototype week 1. Present per-feature values *summed over timesteps*, not a raw `timesteps × features` grid — far more readable. (§7, D-17)
8. **Google Trends may lag rather than lead outbreaks** (people search after news coverage). If it only contributes at lag 0 or negative lags, it is a **nowcasting** signal, not a predictive one. Test it explicitly; a null result is a reportable finding. (§4, §14)

---

## 7. Engineering standards

Set by the user on 2026-08-21. These govern all code in this repo.

| ID | Rule |
|----|------|
| E-01 | Correctness and reuse over volume of code. |
| E-02 | Before writing a function, check whether one in this repo already does the job. If it does, call it. If it nearly does, **generalise it** — never write a second copy. |
| E-03 | Every function does one thing, is type-annotated, and is testable in isolation. |
| E-04 | No magic numbers. Every constant comes from config. |
| E-05 | No code in `__main__` blocks beyond a thin CLI entry point. |
| E-06 | A third similar block means stop and extract an abstraction. |
| E-07 | If an instruction would produce a worse design than a visible alternative, say so **before** coding. |

### Design consequences

**A flat constants file will not survive §6.** The ablation grid (A/B/C data sources × D/E/F feature designs) means the experiment *is* configuration — E-04 forces a composable config, not a `constants.py`. Plan: frozen dataclasses `DataConfig` / `FeatureConfig` / `ModelConfig` composed into `ExperimentConfig`, with the ablation grid generated by varying fields. Each run serialises its own config next to its metrics in `results/`, so every number in the report is reproducible and traceable to the exact configuration that produced it.

**Notebooks must stay thin.** README §12 mandates four notebooks; E-03 and E-05 make them a liability, because logic defined in a cell cannot be tested and silently forks from `src/`. Rule: **notebooks define no functions.** They import from `src/`, call, and display. Otherwise the report figures stop matching the code that ships.

**E-02 needs something to reuse.** The repo is empty, so the near-term risk is the inverse — premature abstraction. The one contract worth fixing up front is a **canonical tidy schema** (`state`, `date`, + value columns) that every loader in `src/data/` returns. With it, fusion is one merge; without it, fusion becomes per-source special-casing and E-06 triggers on the third data source.

---

## 8. Core abstractions

Three abstractions carry the whole project (user, 2026-08-21). Right in Phases 0–4 → the rest is assembly. Wrong → every later phase duplicates work.

| ID | Abstraction | Used by |
|----|-------------|---------|
| A-1 | `build_features(panel, cfg)` — raw panel → model-ready `(X, y)` | training, evaluation, **and scenario simulation** |
| A-2 | `Forecaster` protocol — `.fit(X, y)` / `.predict(X)` | naive baselines, GBM, LSTM — so the ablation is **one loop** |
| A-3 | `run_experiment(model, data, cfg)` → metrics dict + artifacts | every baseline, every ablation, every horizon |
| A-4 | `forecast_horizon(panel, state, target_date, model, cfg)` → `ForecastCurve` | the dashboard's forward projection (added 2026-08-25) |

**A-1 matters most.** Scenario simulation is correct *only* if it re-runs the same feature pipeline on modified raw inputs. If simulation owns a second copy of the feature code, the two drift and every what-if result becomes meaningless. This is D-08 enforced structurally rather than by discipline, and it is the reason A-1 exists.

### Refinements

| ID | Refinement | Reason |
|----|------------|--------|
| A-1a | `build_features` returns a **`FeatureBundle`**, not a bare tuple: `X`, `y`, `feature_names`, and the `(state, date)` index. | A bare `(X, y)` discards the time index that rolling-origin CV splits on (D-04) and the state key the choropleth maps predictions back to (§10). SHAP also needs `feature_names` to report per-feature values summed over timesteps (§7). |
| A-1b | `build_features` performs **stateless, row-local transforms only** — lags, rolling aggregates, spatial lags, cyclic terms, target construction. **Scaling is excluded.** | Lags and rolling means are deterministic given the past and leak nothing. Scaling learns parameters from data, so it must be fold-aware (D-05). Keeping it out means `build_features` runs once and cannot leak. |
| A-1c | The simulator applies the **already-fitted** scaler from the trained model. It never refits. | Refitting on simulated inputs would shift the feature distribution the model was trained against, silently changing predictions for reasons unrelated to the scenario. |
| A-2a | Canonical `X` is 3-D `(n, timesteps, n_features)`. Tabular Forecasters flatten inside their own `.fit` / `.predict`. | Keeps A-2 a single protocol across LSTM and tabular models. Flattening is already required by the SHAP wrapper (S-5), so the reshape logic is shared, not duplicated. |
| A-2b | Intervals come from a **`ConformalForecaster(base: Forecaster)` wrapper** that itself implements `Forecaster` — not from a `.predict_interval` method on the protocol. | Model-agnostic, so intervals and CRPS (S-6) are available for *every* model in the ablation including the naive baselines, which is what makes interval calibration comparable. Adding a method to the protocol would force every baseline to implement it. |
| A-2c | Feature config **must always emit the seasonal lag column** (12 monthly / 52 weekly) even when the ablation excludes it elsewhere. | Seasonal-naive needs the value from the same period last year. If `cfg` caps lags at k=6, that column does not exist, the baseline cannot read it from `X`, and it breaks out of the protocol — destroying the "one loop" property. |
| A-3a | `run_experiment` takes a **model factory** `Callable[[], Forecaster]`, never a constructed instance. | With an instance, fold 2 continues training on fold 1's fitted weights. This produces no error — just silently contaminated CV results, and an ablation table that cannot be reproduced. |
| A-3b | `run_experiment` owns the rolling-origin loop **and** the per-fold scaler fit. | Placing the scaler fit inside the fold loop makes D-05 leakage structurally impossible rather than a rule someone has to remember. |
| A-3c | Returns a typed `ExperimentResult`: per-fold metrics, aggregate mean ± std, predictions, the serialised `cfg`, and model artifacts. | D-04 requires mean ± std across folds, so per-fold values must be retained, not just the aggregate. Serialising `cfg` alongside makes every number in the report traceable (§7). |
| A-3d | `run_experiment` must know the **inverse target transform** and report metrics on the interpretable scale. | `y` is `log(cases_per_100k + 1)` (D-02). MAE on the log scale is meaningless to a public-health reader. Back-transforming via `exp(y)-1` is biased by Jensen's inequality — decide and document whether metrics are reported on log scale, back-transformed scale, or both. **Open — see Q-08.** |
| A-4a | `forecast_horizon` lives in `src/simulate.py` and recurses by **extending the raw panel and re-entering `model.predict`**, never by touching a lag directly. | Same argument as A-1. A fed-back prediction has to move every derived column that depends on cases; going back through `build_features` is the only way to guarantee it does. This is why it is not a new module. |
| A-4b | It is **not** `simulate()`, and both docstrings say so. `simulate` answers *"what if conditions were different?"*; `forecast_horizon` answers *"what happens next under typical conditions?"* | Reporting one as the other either invents a policy claim from a plain forecast, or dresses a counterfactual up as a prediction. |

---

## 9. Data contract — TO BE FILLED IN PHASE 1

| Category | Variables | Candidate source | Granularity | Status |
|----------|-----------|------------------|-------------|--------|
| Epidemiological | Dengue case counts | NVBDCP / IDSP | State, monthly or weekly | ❌ not located |
| Climatic | Rainfall, temperature, humidity | IMD, ERA5 (`cdsapi`) | Daily gridded → state, **population-weighted** (S-7) | ❌ not located |
| Demographic | Population, population density | Census of India | Static per state | ❌ not located |
| Public awareness | Search interest: "dengue", "dengue symptoms", "dengue fever" | `trendspy` / Trends CSV / Wikipedia Pageviews (S-4) | Weekly, state | ❌ not located |

**Feature families planned** (§5.1): point lags `{rainfall, temperature, humidity, cases, trends}_lag_1..k` · rolling aggregates (e.g. 4-period mean rainfall) · spatial lags (adjacency-weighted neighbour cases at t-1, t-2) · static/normaliser (population density → per-100k target) · cyclic seasonality (D-13).

---

## 10. Repo map (planned, §12)

```
MAJOR PROJECT/
├── brain.md                     ← this file
├── README.md                    ← frozen spec
├── RUNNING.md                   ← install and pipeline order (start here to run it)
├── requirements.txt
├── data/{raw,interim,processed}/
├── notebooks/                   01_data_exploration · 02_feature_engineering
│                                03_model_training · 04_shap_analysis
├── src/
│   ├── data/        loaders, cleaners, fusion
│   ├── features/    lag creation, spatial lags, scaling
│   ├── models/      LSTM, baselines, training loop
│   ├── explain/     SHAP wrappers
│   ├── simulate/    scenario engine
│   └── recommend/   decision-support logic
├── dashboard/app.py
└── results/{figures,metrics}/
```

**Roadmap gates (§13):** 1 data verified *(blocking)* → 2 clean unified dataset → 3 features, no leakage → 4 baseline scores recorded → 5 LSTM beats seasonal naive → 6 RQ1+RQ2 answered → 7 SHAP renders *(start in parallel with 5)* → 8 lags move coherently → 9 thresholds data-derived → 10 end-to-end demo → 11 report.

---

## 11. Change Log

Append-only. Newest at the bottom. Format:

```
### YYYY-MM-DD — <short title>
**What:** what changed
**Why:** reasoning / which README section or brain.md ID it serves
**Files:** paths touched
**Follow-up:** anything left open
```

---

### 2026-08-21 — brain.md created
**What:** Read README.md line by line. Created `brain.md` as the project's working memory. Copied `README.md` from Desktop into the project folder as the frozen spec.
**Why:** User requested a persistent reference file that I consult for context and update with every improvement.
**Files:** `brain.md` (new), `README.md` (copied in)
**Follow-up:** Q-01 through Q-07 unanswered; Phase 1 not started.

### 2026-08-21 — Cross-section audit of the README
**What:** Recorded 8 supersessions (§3, S-1..S-8) where §15 overrides §5–§11, and 6 internal contradictions (§5, C-1..C-6) with proposed resolutions.
**Why:** §15 was appended after the main spec and silently invalidates earlier decisions (MC dropout, pytrends, DeepExplainer, per-state models). Following the README top-to-bottom without this map would produce code that contradicts the current plan — e.g. building on `pytrends`, which is archived and broken.
**Improvements added beyond the README:**
- Promoted **spatial lag features** from *Recommended* to **Core** (D-15) — the fixed title says "spatio-temporal", so a temporal-only model is overclaiming, which §14 itself flags as a risk.
- Promoted **prediction intervals** from *Recommended* to **Core** (D-16) — §9's required output format and upper-bound alerting both depend on them.
- Merged the two conflicting target definitions into **`log(cases_per_100k + 1)`** (D-02) — satisfies both the variance-stabilisation requirement (§5.2) and cross-state comparability (§15.2).
- Resolved the pooled-model vs. per-state-feature-selection conflict (C-4): global top-k feature set for the model, per-state SHAP rankings reported as a finding.
**Follow-up:** C-3 (forecast horizon) cannot be closed until Q-01 (data granularity) is answered.

### 2026-08-21 — Engineering standards adopted
**What:** Added §7 Engineering standards (E-01..E-07) and renumbered the sections that followed. No code written.
**Why:** User set repo-wide engineering standards. Recording them here means they survive between sessions and bind all future code.
**Improvements added beyond the README:**
- Ruled that config must be **composable dataclasses**, not a flat constants module — the §6 ablation grid is itself configuration, and each run serialises its config beside its metrics so every reported number is reproducible.
- Ruled that the four README §12 notebooks **define no functions** — they import from `src/` only, so logic cannot fork between notebook and module.
- Identified the **canonical tidy schema** (`state`, `date`, + values) returned by every loader as the one abstraction worth fixing before any code exists, so fusion stays a single merge.
**Files:** `brain.md`
**Follow-up:** Q-01 still blocks Phase 1. The SHAP prototype (D-17) is the only work item not blocked by it.

### 2026-08-21 — Core abstractions defined
**What:** Added §8 Core abstractions (A-1 `build_features`, A-2 `Forecaster` protocol, A-3 `run_experiment`) from the user, plus ten refinements. Renumbered following sections. Added Q-08 and Q-09. No code written.
**Why:** These three signatures determine whether later phases assemble or duplicate. A-1 in particular enforces D-08 structurally: one feature pipeline shared by training and simulation, so what-if results cannot drift from training behaviour.
**Improvements added beyond the user's proposal:**
- **A-3a — `run_experiment` must take a model *factory*, not an instance.** With an instance, each CV fold continues training on the previous fold's fitted weights. It raises no error; it silently contaminates every ablation number. This is the highest-severity issue found.
- **A-1a — return a `FeatureBundle`, not a bare `(X, y)`.** A bare tuple discards the `(state, date)` index that rolling-origin CV splits on and that the choropleth needs to map predictions back to states, plus the `feature_names` SHAP requires.
- **A-1b / A-3b — scaling belongs in `run_experiment`'s fold loop, not in `build_features`.** Splitting stateless transforms (leak-free by construction) from fitted ones makes D-05 leakage structurally impossible instead of a rule to remember.
- **A-2b — conformal intervals as a `Forecaster`-wrapping decorator**, not a protocol method. Gives intervals and CRPS for every model including the naive baselines, which is what makes calibration comparable; a protocol method would force each baseline to implement it.
- **A-2c — config must always emit the seasonal lag column.** Seasonal-naive reads the same period last year; if `cfg` caps lags below 12/52 that column does not exist and the baseline falls out of the protocol, breaking the one-loop property.
- **A-1c** simulator reuses the fitted scaler; **A-2a** canonical 3-D `X` with tabular models flattening internally; **A-3c** typed `ExperimentResult` retaining per-fold metrics for D-04's mean ± std; **A-3d** metrics reported on an interpretable scale (raised as Q-08).
**Files:** `brain.md`
**Follow-up:** Q-08 and Q-09 open. Q-01 still blocks Phase 1.

### 2026-08-21 — Phase 0: scaffold and contracts
**What:** Built the project skeleton. `config.yaml` + `src/config.py` (frozen validated tree), `src/io.py` (the single `@cached` decorator), `src/artifacts.py` (`save_run`/`load_run`), `DataSource` and `Forecaster` Protocols, docstring-only stubs for features/splits/evaluate/explain/simulate/recommend, `pyproject.toml`, `.gitignore`, pytest config, one smoke test. No data loading, features or models, per the phase boundary.
**Why:** Every later phase plugs into these three contracts. Getting them right now is what stops Phases 2–10 duplicating work.
**Verified:** `pytest` — 1 passed. Manually exercised beyond the smoke test: cache hit/miss counting, keyword-vs-positional-vs-default argument equivalence, distinct arguments not colliding, rejection of unfingerprintable arguments and non-DataFrame returns; artifact round-trip across DataFrame/Series/ndarray/dict, overwrite guard, path-traversal guard; and eight config validation failures each producing a keyed error message. Verification output directories were removed afterwards.
**Improvements added beyond the brief:**
- **`@cached(key)` fingerprints the bound arguments, not just the key.** A bare key would make `fetch_cases("Kerala")` and `fetch_cases("Odisha")` share one file and silently return the wrong state's data. Files are `key__<blake2b>.parquet`; defaults are applied before hashing so positional, keyword and default call forms map to the same entry.
- **`@cached` refuses arguments it cannot fingerprint reproducibly** (DataFrames, arrays). `repr` on those is unstable across processes and can collide for unequal inputs — the failure would surface as a wrong-data bug, not an error.
- **Unknown keys in `config.yaml` are a startup error**, not just missing ones. A typo like `granularty:` would otherwise silently fall back to a default.
- **One generic `_build`/`_coerce` pair** constructs all 13 config sections by type introspection, rather than 13 hand-written parsers (E-06). Rejects `bool` where `int` is declared, since `bool` subclasses `int`.
- **Cross-section validation enforces A-2c and A-3d in code**: `features.lags` must contain `seasonal_period`, and no horizon may exceed `split.test_size`.
- **`manifest.json` per run** — `load_run` reconstructs from a declaration rather than guessing from extensions, and writer/reader dispatch lives in one table so the two cannot drift.
- **`artifacts.py` imports no ML framework.** Objects with `.save()` delegate to it and load back as a `Path`, so the dashboard can read runs without TensorFlow installed. TF and SHAP are in `[project.optional-dependencies] model` for the same reason.
- **Protocols carry the contract in their docstrings**: 3-D canonical `X` with tabular models flattening internally (A-2a), no scaling inside models (A-1b), factories not instances (A-3a), canonical tidy panel with `PANEL_KEYS` (A-1a), and loaders leaving gaps as NaN rather than imputing.
**Deviation noted:** `src/` is the import root, so modules import as `from src.config import ...`. Conventional layout would be `src/<package>/`. Implemented as specified; flagged only because it makes the project awkward to `pip install` as a library, which does not matter for an application.
**Files:** `config.yaml`, `pyproject.toml`, `.gitignore`, `src/*.py`, `src/sources/__init__.py`, `src/models/__init__.py`, `tests/test_config.py`, `dashboard/.gitkeep`
**Follow-up:** Q-01 still blocks Phase 1. Q-05 (states) and Q-09 (validation split) are stubbed in `config.yaml` with comments and will need real values. `dashboard/app.py` arrives in Phase 10.

### 2026-08-21 — Phase 1: data acquisition
**What:** `src/sources/registry.py` (canonical states, aliases, boundary changes, adjacency), `src/sources/base.py` (`BaseDataSource` + auto-discovery + the cached pipeline), four sources (`cases`, `climate`, `awareness`, `demographic`), `src/panel.py` (`assemble_panel`, `data_quality_report`, `summarise_panel`). Added `quality:` and `preprocess:` config sections. Generalised `io.cached` with `key_args` so one decorator serves all sources with readable filenames.
**Why:** Every source behind one interface, cached, normalised through one registry.
**Verified:** 68 tests pass. Auto-discovery checked empirically by dropping a new source file into the package and resolving it without editing anything else.
**Improvements added beyond the brief:**
- **Boundary changes are modelled, not aliased away.** Telangana split from Andhra Pradesh in **June 2014** — inside the configured 2010–2023 window. Pre-2014 Andhra Pradesh counts and population include Telangana, so the series measures two different territories under one label. No alias map can fix this; `BOUNDARY_CHANGES` records it and `summarise_panel` reports affected states. Same for J&K/Ladakh (2019) and the DNH/DD merger (2020).
- **Adjacency defined one-directionally and symmetrised on load**, so a missing reverse entry cannot create a directed spatial-lag graph. Tested for symmetry, canonical names, and no self-loops.
- **Auto-discovery via `__init_subclass__` + `pkgutil`** — adding a source is genuinely one file, with no registry list, import block or dispatch table to edit.
- **Normalise before filtering.** Filtering to the study states first would drop a source spelling Odisha as "Orissa" while the config asked for "Odisha".
- **Duplicate `(state, date, variable)` rows are rejected before the pivot.** `pivot_table` would silently keep one, producing a panel that is quietly wrong.
- **Awareness backend honesty:** Wikipedia Pageviews has **no Indian-state granularity**, so that fallback broadcasts one national series to every state and contributes zero between-state variation to a pooled model. Documented in the module and surfaced by which backend ran, rather than left for a reviewer to find.
- **Population interpolated geometrically between census anchors**, held flat outside them rather than extrapolated. Freezing one census year across 14 years would distort the per-100k target.
**Files:** `src/sources/{registry,base,cases,climate,trends,static,__init__}.py`, `src/panel.py`, `src/io.py`, `src/config.py`, `config.yaml`, `tests/{conftest,test_registry,test_sources,test_panel}.py`
**Follow-up:** Q-01 unanswered — no raw data obtained, so the reality check (monthly vs weekly, points per state) is still open. `summarise_panel` computes it automatically the moment data lands.

### 2026-08-21 — Phase 2: preprocessing
**What:** `src/preprocess.py` — `reindex_complete`, `detect_outliers`, `find_long_gaps`, `interpolate_short_gaps`, `flag_quality`, composed by `preprocess(panel, cfg)` returning a `PreprocessResult`.
**Why:** A clean panel on a complete regular index, with every fitted operation deferred to the fold loop.
**Review gates:**
- *Does anything call `.fit()`?* No — enforced by a test that greps the module body for `.fit(`, `.fit_transform(`, `Scaler(`, `Imputer(`.
- *Anything imputed that should have been reported?* Case counts are never interpolated — only climate, and only gaps ≤ 2 periods. Everything else stays NaN and is listed in `long_gaps`.
**Two real bugs found and fixed by the tests:**
- **`mad_outliers` was blind to the outbreaks it existed to catch.** The MAD collapses to zero whenever more than half the observations are identical — routine in dengue counts, where a state can report the same low number or zero for most of the year. The zero-MAD guard then returned "no outliers" for a series of thirty zeros and one 250-case month. Now falls back to the mean absolute deviation, which stays positive as long as any two values differ.
- **`interpolate_short_gaps` was partially filling long gaps.** pandas' `limit=` caps *how many consecutive values are filled*, not the gap size it will attempt — so a five-month hole under `limit=2` came back with its first two months invented and the rest missing. That is worse than either alternative: the series looks continuous where it is not. Runs are now measured first and over-long ones masked back out.
**Other decisions:** interpolation runs per state (never across the panel); `limit_area="inside"` so leading/trailing gaps are never back-filled; outliers flagged and kept, since in a dengue series the extremes *are* the events being predicted; detection is per state, so a high-baseline state does not flag its whole series.
**Files:** `src/preprocess.py`, `config.yaml`, `src/config.py`, `tests/test_panel.py`
**Follow-up:** none blocking. Phase 3 (features) still needs Q-01.

### 2026-08-21 — Phase 3: feature engineering
**What:** `src/features.py` — `build_features(panel, cfg) -> (X, y, FeatureSpec)`, composing `add_identity`, `add_lags`, `add_rolling`, `add_spatial_lags`, `add_cyclic`, `add_static`, `build_target`, `window_sequences`, plus `flatten`. `FeatureSpec` carries column order, per-column `FeatureOrigin` provenance, and the `(state, date)` sample index.
**Why:** A-1 realised. One builder shared by training, evaluation and simulation, so what-if results cannot drift from training behaviour.
**Verified:** 96 tests pass. Purity checked three ways — determinism, non-mutation of the input panel, and call-order independence — plus a source grep banning `.fit(`, `Scaler(`, `Encoder(`, `lru_cache`.
**Bug found and fixed during Phase 3 — a config-level design error, not a typo:**
- **`sources_for()` was ablating at fetch time.** It selected sources from `features.sources`, so ablation A (climate only) would never have loaded cases or population — and the target is built from exactly those. Configuration A would have had nothing to predict. Split into `data.sources` (what gets loaded) and `features.sources` (what feeds the model), with a cross-section validation that the latter is a subset of the former.
- **`population` was entering the design matrix as a feature** while also being the target's denominator, and `population_density` was built twice (once as a selected variable, once as a static). Fixed by giving each transformation its own explicit config list (`level_variables`, `lag_variables`, `rolling_variables`, `static_variables`), with a validation rejecting overlap between level and static.
**Design decisions worth remembering:**
- **`sample_index` dates are the forecast origin**, the last observed period in the window — not the period being predicted. Rolling-origin splitting depends on reading it that way.
- **Spatial lags yield NaN, not zero,** for a state whose neighbours are outside the study. Zero would assert neighbouring counts were *observed* to be zero.
- **`spatial_weight_scheme: uniform` is the default** because `population` weighting uses each state's mean population across the whole window, including future test folds. Slowly-varying and not target-derived, so the risk is small — but not zero, hence not the default.
- **Incomplete windows are dropped, never padded.** Padding teaches the model to reproduce padding.
- Rolling windows use `min_periods == window`, so a partial window is never a different statistic wearing the same column name.
**Open concern — feature dimensionality (measured, not estimated).** Under the shipped config with 12 states × 168 monthly periods: `X = (1728, 12, 51)` → **612 flat inputs per sample against 1728 samples, a ratio of 2.8**. Lag features and a 12-step sequence window encode the same history twice, and the reference paper's failure mode was exactly this kind of over-capacity. Measured alternatives, both config-only: `sequence_length: 1` gives 51 inputs (ratio 36.5); `include_lags: false` gives 120 (ratio 15.4). No code change needed either way — but the decision should be made before Phase 5, not after the LSTM loses to seasonal-naive. **Raised as Q-10.**
**Files:** `src/features.py`, `src/config.py`, `config.yaml`, `src/sources/base.py`, `tests/test_features.py`
**Follow-up:** Q-10 open. Q-01 still unanswered.

### 2026-08-21 — Audit of Phases 0–3
**What:** Full review of everything built so far. `ruff` clean, `mypy --disallow-untyped-defs` clean on 20 source files, 103 tests passing, zero warnings.
**Why:** User asked for a correctness check and optimisation pass before continuing.
**Two test gaps found and closed — these were the real findings:**
- **No leakage test existed.** Added one that overwrites the entire panel from a cut date forward with 1e6 and asserts every sample whose forecast origin precedes the cut is bit-identical. Stronger than checking shifts individually, because it catches any path future data could take — rolling windows, the spatial graph, the target shift — without enumerating them. **It passes**: 1272 of 1836 past-origin samples unchanged. The companion assertion is that `y` *does* move for origins within one horizon of the cut, since otherwise the target would not be a forecast.
- **`assemble_panel` itself was untested.** Only its helpers were. Added `tests/test_assemble.py`: fusion on `(state, date)`, complete-grid reindexing, partial coverage staying NaN, duplicate-variable rejection, empty-input rejection, and sorted output (which `build_features` requires).
**Two typing holes found by mypy, both real:**
- `config._coerce` called `_build` on anything `is_dataclass()` accepted, which includes dataclass *instances*, not just classes. Narrowed with `isinstance(hint, type)`.
- `trends.parse` used `# type: ignore` to paper over an untyped backend payload. Replaced with an `_expect()` narrowing helper that fails naming the backend, rather than surfacing as an `AttributeError` three frames away.
**Performance — measured, and deliberately left mostly alone.** Profiled at the largest plausible scale (36 states × 730 weekly periods = 26,280 rows): `preprocess` 0.24s, `build_features` 0.23s, quality report 0.12s. Nothing is slow enough to justify restructuring. One change made because it was also the clearest hotspot and simpler code: `detect_outliers` replaced a nested `.loc` assignment loop with a `groupby.transform`, **0.24s → 0.17s** on `preprocess` overall. `window_sequences` alignment also switched from `target.loc[state]` to `target.reindex(group.index)` — same speed, but the two can no longer drift if groupby order and target order ever differ.
**Also:** 30 ruff autofixes (modern `collections.abc` imports, `functools.cache`, `zip(strict=True)`), 18 over-long lines wrapped by hand rather than by raising the limit, and per-file lint ignores so test-name-as-documentation does not trip the docstring rule.
**Files:** `src/config.py`, `src/features.py`, `src/preprocess.py`, `src/io.py`, `src/artifacts.py`, `src/sources/*.py`, `pyproject.toml`, `tests/test_assemble.py`, `tests/test_features.py`, `tests/test_sources.py`
**Follow-up:** Q-01 and Q-10 still open. No known defects outstanding.

### 2026-08-21 — Phase 4: evaluation harness and baselines
**What:** `src/splits.py` (`rolling_origin`, `Fold`), `src/evaluate.py` (`run_experiment`, `compute_metrics`, `RunResult`, `compare`), `src/models/scaling.py` (`StandardScaled`), `src/models/naive.py` (`LastValue`, `GBMBaseline`, factories), `scripts/run_baselines.py`. Added `target_level` autoregressive features, which the baselines read their predictions from.
**Why:** Know what "good" means before building anything complex.
**Verified:** 128 tests pass, ruff and mypy clean. All three baselines ran through one unmodified `run_experiment` and are saved in `results/runs/`.

**Critical finding — the specified `splits.py` signature was unsafe, and I changed it.**
`rolling_origin(n_samples, cfg)` cannot produce a temporal split on a pooled panel. Samples are ordered state-then-date, so a positional cut holds out **states, not time**. Measured on the real config: an 80/20 positional split gave train and test both spanning 2011-03 → 2023-11, separated only by state. That would have been reported as forecasting accuracy while actually measuring cross-state generalisation, with the model having seen the entire test period. The signature is now `rolling_origin(sample_index, cfg)`, cutting on dates; passing an int raises with an explanation. **Tested**: every state must appear on both sides of every cut, and `train_dates.max() < test_dates.min()`.

**Second correctness addition — embargo.** A training sample at origin `T` carries a label from `T + horizon`. Without a gap, the last training labels come from inside the test window. Each fold now drops `horizon` periods between val and test. Fold layout is `[fit | val | embargo | test]`.

**Deviation from the brief — scaling lives in a wrapper, not in `run_experiment`.** The spec said the harness should fit the scaler. That breaks the naive baselines: persistence predicts by reading the current value out of its input, and if the harness has already standardised that column it reads a z-score, not a case rate. Making the harness scale for some models and not others requires branching on model type — precisely what review gate 1 forbids. `StandardScaled(base)` is itself a `Forecaster`, so each factory composes it or not. The fold guarantee is unchanged and arguably stronger: the wrapper only ever sees one fold's training rows.

**Q-09 resolved.** Validation is now an explicit block from `splits.py` (tail of train, time-ordered), not carved inside the model. It also serves as the conformal calibration set in Phase 6.

**Review gates:**
- *Any `isinstance(model, LSTM)`?* No — enforced by a test grepping `src/evaluate.py` for `isinstance(model`, `type(model)`, `LSTM`, `GBMBaseline`, `LastValue`.
- *Scaler fitted inside the fold?* Verified as asked: per-fold scaler means are collected and asserted to differ across folds, plus a test that fitting on `fold.fit` differs from fitting on everything.
- *Seasonal-naive MAE?* **See below — recorded, but synthetic.**

**Seasonal-naive uses lag `P − h`, not lag `P`.** Predicting `y(t+h)` with "same period last year" means `y(t+h−P)`, which from origin `t` is lag `P−h`. Lag `P` would predict a year before the *origin* — a different, weaker baseline. These lags are emitted automatically so the baseline works whatever driver lags are configured.

**Baseline scores — SYNTHETIC PANEL, not dengue.** No real data exists yet, so `--synthetic` generates a seasonal panel to exercise the harness. Numbers describe the generator; runs are prefixed `synthetic_` so they cannot be mistaken for results. 12 states × 168 months, X=(1728, 12, 60), horizon 1, 4 folds:

| run | MAE (cases/100k) | RMSE | R² (log) |
|-----|------------------|------|----------|
| gbm | 0.0236 ± 0.0015 | 0.0294 | 0.958 |
| seasonal_naive | 0.0287 ± 0.0010 | 0.0359 | 0.937 |
| persistence | 0.0701 ± 0.0015 | 0.0823 | 0.670 |

Ordering is the sanity check that matters: on strongly seasonal data seasonal-naive beats persistence 2.4×, and GBM edges ahead of both. **The real bar cannot be set until Q-01 is answered.**

**Also:** metrics reported on both scales (Q-08 — `*_log` for model comparison, `*_cases_per_100k` for the report, with the Jensen bias of `expm1` documented); R² is NaN not 0.0 on a constant window; CRPS is computed now and equals MAE for point forecasts, which is exactly the bar Phase 6 intervals must beat.
**Files:** `src/splits.py`, `src/evaluate.py`, `src/models/{scaling,naive}.py`, `src/features.py`, `src/config.py`, `config.yaml`, `scripts/run_baselines.py`, `tests/test_evaluate.py`
**Follow-up:** Q-01 and Q-10 open. Phase 5 (LSTM) is unblocked for code, but its scores are meaningless until real data lands.

### 2026-08-23 — Evaluated Alka Rani's handoff model
**What:** Independently evaluated `Desktop/dengue_model_handoff`. Loaded the model, reproduced her reported metrics exactly, recovered the test window forensically, benchmarked against naive baselines.
**Verification:** Wrote an exact numpy LSTM forward pass from the saved weights (TF was broken mid-repair). Matches TensorFlow to 2.5e-08 and reproduces her README numbers exactly (MAE 0.2141, RMSE 0.7377, R2 -0.0779), so the analysis below is solid.

**What the model is:** Sequential, 8 weeks x 8 features. LSTM(64, seq) -> Dropout(0.2) -> LSTM(32) -> Dropout(0.2) -> Dense(16, relu) -> Dense(1). Keras 3.15.1, saved 2026-08-22. Features: dengue_cases_log, RAIN, RH03, RH12, TMAX, TMIN, week_sin, week_cos.

**Data (answers Q-01, but not in the shape we assumed):** WEEKLY, BLOCK level, Uttar Pradesh only. 75 districts, 829 blocks, 858 location_ids, 65,222 rows, 2023-06-19 to 2025-08-11. Roughly 26 months, so two dengue seasons. 81.6% of all weeks are zero cases.

**The model is better than its own README suggests.** The README reports count-scale metrics, where R2 is negative and looks alarming. On the log scale the model actually trains on:

| model | MAE | RMSE | R2 |
|---|---|---|---|
| LSTM (improved) | 0.1401 | **0.2558** | **+0.1073** |
| always zero | **0.0830** | 0.2832 | -0.0939 |
| always mean | 0.1497 | 0.2707 | 0.0000 |
| persistence | 0.1026 | 0.3034 | -0.2561 |

Best RMSE and best R2. It loses on MAE only to always-zero, expected when 90% of targets are zero. On the 1,133 non-zero weeks it beats every baseline: MAE 0.6462 vs 0.7197 persistence, 0.7633 mean, 0.8462 zero. Correlation with actuals 0.3815. Real signal, modest size. Her model-selection rationale was sound.

**The real problem: the test window is the dengue off-season.** Recovered test dates by unscaling the last timestep and matching climate fingerprints back to the CSV (6,504 of 11,556 matched uniquely). Test origins run **2024-12-30 to 2025-04-28**. Those months hold **3.4%** of all dengue cases; September to November hold **91.2%**. The model is graded almost entirely on predicting zeros and is barely tested on outbreak conditions. The split is at least time-ordered, so no random-split leakage.

**Other issues:** README names `best_lstm_model.keras` and `weighted_mae_lstm_model.keras` but only `improved_lstm_model.keras` is present. The metric table mixes scales without saying so. `requirements.txt` is unpinned, and **TensorFlow 2.21 is broken on this Windows box** (tflite DLL import failure); 2.20.0 works.

**Granularity mismatch with our project:** ours is state-level monthly India-wide; hers is block-level weekly UP-only. The model cannot drop into our pipeline as-is, and the feature sets differ (she has RH03/RH12; we have derived humidity, spatial lags, target lags). Using her data would mean re-deciding Q-01 and Q-05.
**Files:** none changed in this repo. Environment now: tensorflow 2.20.0, keras 3.15.1, scikit-learn 1.9.0, protobuf 7.36.0.
**Follow-up:** asked Alka for the other two model files, the training script, any pre-June-2023 data, and a re-split with a Sep-Nov test window.

### 2026-08-23 — Phase 5: pooled LSTM
**What:** `src/models/lstm.py` (`PooledLSTM`, `pooled_lstm` factory), state-identity features, `scripts/run_lstm.py`. Extended `Forecaster.fit` with an optional validation block and threaded it through the harness.
**Verified:** 143 tests pass, ruff and mypy clean. LSTM runs through `run_experiment` with no special-casing.

**Architecture:** sequence columns -> LSTM stack (48 units, 2 layers, dropout 0.2) -> concat with state embedding and static features -> Dense -> Dense(1). MSE on `log(cases_per_100k + 1)`. Adam, early stopping patience 10, max 100 epochs, all from config. `keras.utils.set_random_seed` for determinism. 
State identity is one-hot columns in `X`, read at the forecast origin and projected through a **bias-free Dense layer**, which is mathematically an embedding lookup. Doing it this way keeps `Forecaster.fit` a two-argument call instead of needing a second input tensor, and lets the tree baselines split on state too.

**Harness change, as invited by the brief.** `Forecaster.fit` now takes an optional `validation` block, passed uniformly to every model. The LSTM needs a validation set for early stopping and **cannot carve it from the tail of `X`**: `X` is ordered state-major, so its tail is one *state*, not one time *period*. An internal carve would have early-stopped on a held-out state while claiming to hold out time. `src/splits.py` already cuts a proper time block, so the harness hands it over. Baselines accept and ignore it; no branch on model type anywhere.

**New trap found and closed — states can vanish silently.** A state whose neighbours are all outside the study gets an all-NaN spatial lag, so every one of its windows is incomplete and it drops out of the dataset entirely. Worse, that makes the spatial ablation unfair: configs E and F would be scored on **different sets of states**. `FeatureSpec.dropped_states` now records it, with a test. The shipped 12-state config is unaffected (every state has an in-study neighbour), but the 3-state test fixture loses Odisha.

**Bug fixed in the artifact store:** `save_run(overwrite=True)` raised `PermissionError` on `rmdir` because OneDrive holds a handle on synced directories. Now clears contents and reuses the directory.

**Review gates (on SYNTHETIC data - see caveat):**
| run | MAE (cases/100k) | RMSE | R2 (log) |
|---|---|---|---|
| gbm | 0.0235 | 0.0295 | 0.958 |
| **lstm** | **0.0273** | 0.0346 | 0.942 |
| seasonal_naive | 0.0287 | 0.0359 | 0.937 |
| persistence | 0.0701 | 0.0823 | 0.670 |

- *Does the LSTM beat seasonal naive?* On synthetic data yes, 0.0273 vs 0.0287. **This does not answer the gate.** The synthetic generator is a clean sine wave; the verdict describes the generator, not dengue. GBM still wins, which is itself worth noting.
- *Is the train/validation gap widening?* No. Per-fold gap is **negative** (val loss below train loss) across all four folds, which is expected with dropout active during training only. Early stopping fired in every fold (best epoch 14/20/19/60 of 24/30/29/70). No shrinking needed yet.
- *A second training loop?* None. Enforced by a test grepping `src/models/lstm.py` for `rolling_origin`, `for fold`, `compute_metrics`.

**Files:** `src/models/lstm.py`, `src/models/{__init__,naive,scaling}.py`, `src/evaluate.py`, `src/features.py`, `src/artifacts.py`, `src/config.py`, `config.yaml`, `scripts/run_lstm.py`, `tests/test_lstm.py`
**Follow-up:** the real gate cannot be answered until real data in our schema exists. Q-01, Q-05 and Q-10 all now interact with the decision about whether to adopt Alka's weekly UP data.

### 2026-08-23 — Phase 6: ablations and prediction intervals
**What:** `src/uncertainty.py` (`ConformalForecaster`, `predict_interval`, `coverage`, `crps_from_quantiles`), `src/experiments.py` (`run_ablations`, `significance`), `scripts/run_ablations.py` with a comparison plot, `experiments:` grid in `config.yaml`. CRPS, coverage and interval width added to the harness metrics.
**Verified:** 164 tests, ruff and mypy clean.

**Bug found and fixed — the conformal calibration window was calibrating on one state.** `calibration_window: 24` truncated to the last 24 **rows** of the validation block. Rows are ordered state-major, so that is the last 24 periods of whichever state sorts last, not the last 24 periods across all twelve. Measured effect: coverage **72.7% ± 10.3%** against a nominal 80%, with one fold at 59%. Every alert built on the upper bound would have fired late. This is the same state-major trap that bit `splits.py` in Phase 4 and early stopping in Phase 5 — a pooled panel makes "the last N rows" mean something different from "the most recent N periods", and it is worth checking anywhere that phrase appears.
Fixed by using the whole validation block; the rolling-window property already comes from the fold structure, since `splits.py` slides the validation block forward with each fold. `conformal.calibration_window` is now `min_calibration_residuals`, a floor below which intervals are refused. **Coverage after the fix: 80.7% ± 3.7%** on 156 to 228 residuals per fold.

**Config error the geometry check caught:** `initial_train_size: 96` fitted horizon 1 but not horizon 3, which loses two more usable origins to the target shift plus embargo. The whole grid failed with a clear message rather than silently running fewer folds. Reduced to 90. **The split geometry must fit the longest horizon, not the shortest.**

**Ablation grid** is six named configurations in `config.yaml` (A/B/C sources, D/E/F feature design) x two horizons x four models, run by one loop. Adding a configuration is a YAML edit. Two combinations are legitimately unavailable: climate-only has no case history, so persistence and seasonal-naive cannot be constructed. Recorded as skips with reasons rather than silently omitted.

**Review gate 1 - does the table support a claim?** On the baseline grid, **no**. Zero of 32 comparisons exceeded one fold standard deviation. The runner says so itself: "Everything is within noise. Report a null result, not a ranking." Also worth stating in the write-up: the naive baselines are **identical across every feature configuration**, because they read only the target's own lags. That is correct, not a bug, and it means the A-F ablations only ever test the *learned* models.

**Review gate 2 - are intervals calibrated?** Yes, after the fix: 80.7% ± 3.7% against nominal 80%, per-fold 79.9 / 76.4 / 85.4 / 81.2.

**Full grid results (44 runs, synthetic):** 1 of 44 comparisons exceeded one fold std, the LSTM at h1 with climate only vs full+spatial (gap 1.28 std). Everything else is noise. LSTM coverage 80.9% (range 79.9-82.6) against nominal 80%. Two things worth carrying into the real run: **lag features are not earning their place** (D-without-lags is nominally *better* than E-with-lags at horizon 3 and for GBM, while multiplying features from 29 to 70, which is Q-10 showing up empirically), and **persistence collapses at horizon 3** (R2 +0.67 to -1.01) while seasonal-naive barely moves, confirming 1-step-ahead is close to trivial.
**Files:** `src/uncertainty.py`, `src/experiments.py`, `src/evaluate.py`, `src/config.py`, `src/models/naive.py`, `config.yaml`, `scripts/run_ablations.py`, `tests/test_uncertainty.py`, `results/FINDINGS.md`
**Follow-up:** `results/FINDINGS.md` written. Every accuracy number in it describes the generator, not dengue. Q-01 still open; Q-10 now has evidence pointing at dropping the lag features.

### 2026-08-24 — Phase 6b: frozen production model
**What:** `src/production.py` (`train_production`, `load_production`, `ProductionModel`, `select_configuration`, `restore_config`), `scripts/freeze_production.py`, `production:` config section, `FeatureSpec.to_dict/from_dict`. Replaced the LSTM's Keras `Lambda` layers with registered `ColumnSelect` / `LastTimestep` layers.
**Verified:** 182 tests, ruff and mypy clean. Artifact frozen, reloaded, and used to predict from a raw panel.

**The artifact:** `results/runs/production/` holding `model.keras`, `scaler.json`, `feature_spec.json`, `residuals.npy`, `config.json`, `trained_at.json`. `load_production().predict(panel)` goes raw panel to forecasts with intervals in one call. **This is the only model Phases 7-10 may load.**

**Selection chose 29 features over 72.** The parsimony tie-break picked `D_without_lags` over `F_lags_and_spatial`: within one fold std of the best, take the fewest features. That is the Q-10 finding applied automatically rather than argued about — a nominal win that is statistically indistinguishable is not a reason to ship 2.5x the inputs.

**Deviation from "refit on the complete dataset", and it is not optional.** The final `production.calibration_periods` (12) periods are held out of the fit to calibrate the conformal intervals. Calibrating on data the model trained on gives intervals far too narrow, and D-11 alerts on the upper bound — an overconfident upper bound means alerts that fire late. 144 calibration residuals in the current artifact.

**Three bugs found while building this:**
- **Keras `Lambda` layers closed over `self`**, so the model could not be serialised at all. Replaced with `@keras.saving.register_keras_serializable` layers carrying `get_config`. The `.keras` file is now standalone. They must be registered before loading, so `load_production` calls `register_serializable_layers()` first.
- **`load_production` read the live `config.yaml`**, not the config the artifact was trained under. The winning run had `include_lags: false` while the file said true, so the panel built 70 features for a 29-feature model. Caught by my own column guard.
- **The fix to that was itself incomplete.** I pinned a hand-picked subset of feature fields and missed `lags` — which changes the column set even when `include_lags` is false, because the target's autoregressive terms derive from it. Now the **whole** `FeatureConfig` travels with the artifact. Paths still come from the live config, since they describe the checkout rather than the model.

**Review gates:**
- *Self-contained?* Yes. `load_run("production")` returns model, scaler, feature spec, residuals, config, metadata; `predict()` needs no other file.
- *Phases 7-10 free of constructors and `.fit()`?* Yes, enforced by an **AST-parsing test**, not a grep: it walks `src/{explain,simulate,recommend}.py` and `dashboard/**/*.py` rejecting any `.fit()` call, any import of a model constructor, and any reference to one. A companion test proves the guard actually fires on a deliberate violation.
- *Does re-running replace what everything else reads?* Yes, tested: freeze `small`, assert `load_production()` sees it, freeze `big`, assert it now sees that instead, with one directory replaced rather than accumulated.

**Files:** `src/production.py`, `src/features.py`, `src/models/lstm.py`, `src/config.py`, `config.yaml`, `scripts/freeze_production.py`, `tests/test_production.py`
**Follow-up:** the frozen artifact is built on synthetic data, so it is a rehearsal. Re-run `scripts/freeze_production.py` once real data lands. Q-01 still open.

### 2026-08-24 — Reviewed Alka's GitHub repo (supersedes the handoff-folder verdict)
**What:** Examined `github.com/Alkaaaa11/dengue-outbreak-prediction`, including `notebooks/seasonal_training.ipynb` which is NOT in the handoff folder she sent over WhatsApp.
**Headline: the repo is far better than the folder. She has already fixed the off-season test window I was going to raise.**

**Her season-aware split** (`seasonal_training.ipynb` cell 10) is exactly right:
- TRAIN: to 2024-08-31 (41,347 rows, 15.9% non-zero)
- OUTBREAK VALIDATION: 2024-09-01 to 2024-11-30 (9,884 rows, **44.2% non-zero**) — the real dengue season
- FUTURE HOLDOUT: from 2024-12-01 (13,991 rows, 7.6% non-zero)

**Honest results on the outbreak season** (model trained only on pre-Sep-2024 data):

| model | MAE | RMSE | R2 |
|---|---|---|---|
| **LSTM** | **2.196** | **12.97** | **+0.652** |
| Persistence | 2.496 | 13.93 | +0.599 |
| Predict zero | 3.059 | 22.20 | -0.019 |
| Seasonal naive | 8.614 | 22.97 | -0.092 |

Future holdout: LSTM R2 +0.645, MAE 2.24. So the model genuinely works, and the negative R2 in the handoff README was an artifact of the off-season test window, not the model.

**But it fails at the one thing the project exists for.** Cell 182, on the future holdout:
- All outbreak weeks (61): MAE 7.51, mean actual 8.03, mean predicted 3.57
- **Genuine sudden outbreaks (32): mean actual 7.28, mean predicted 1.09** — it under-predicts by ~85%
- Continuing/gradual outbreaks (29): mean actual 8.86, predicted 6.32 (tracks these fine)

It follows outbreaks already underway and misses ones that start. Her Random Forest outbreak classifier is also weak: precision 0.145, recall 0.590, F1 0.233, catching 14/34 sudden outbreaks — roughly 6 false alarms per true one. The "high-confidence" variant catches 1/34.

**Unverified number:** `notebooks/seasonal_regression_results.csv` gives R2 0.867 on 5,715 rows. That row count matches no split in the notebook (season is 9,884, holdout 13,991) and no `to_csv` call produces it in any notebook I could read. **Do not report 0.867** without establishing what it is; the defensible numbers are 0.652 (outbreak season) and 0.645 (future holdout).

**Verdict:** do not retrain from scratch. The architecture, split and evaluation are sound. What needs work is outbreak *detection*, not point forecasting — the loss/target for rare spikes, and alerting built on the interval upper bound plus classifier rather than the point forecast (which is D-11's argument, already in our design).

**Still true:** her data is block-level weekly UP; ours is state-level monthly India-wide. Adopting her model means changing this project's scope, which is a decision for the user and their guide, not a technical detail.
**Files:** none changed. Notebooks cached under `/tmp/alka/`.

### 2026-08-24 — Modification scope for Alka's model (evidence from her dataset)
**What:** Measured the headroom in her model rather than guessing at it. No code changed.
**Key measurements on `weekly_dengue_prepared.csv` (65,222 rows, 858 blocks):**
- **Rainfall leads cases by ~8 weeks.** corr(cases_t, RAIN_t-k): 0.084 at k=0, 0.353 at k=4, **0.508 at k=8**, 0.498 at k=10, 0.443 at k=12. Her input window is **8 weeks**, so it ends exactly where the signal peaks and misses the whole tail. Extending to 16 weeks is the best-evidenced single change.
- **16.0% of log-case variance is between blocks**, and the model has no block identity or embedding. A per-block intercept could absorb it.
- **Top 10% of blocks hold 67.4% of all cases**; 22 of 858 blocks never record one. No population column exists in the dataset, so raw counts treat a large and a small block alike.
- **The 14 unused prepared lag columns are NOT a real gap** — `cases_lag_1..4` and `RAIN_lag_1..2` all sit inside the 8-week sequence window already. Worth not raising as a finding.

**Ranked modification scope:**
1. Sequence 8 -> 16 weeks (cheap, best evidence)
2. Quantile/pinball loss or a hurdle model instead of MSE on log1p — MSE targets the conditional median, and with 65-84% zeros that median is ~0, which is why sudden outbreaks come out at 1.09 against an actual 7.28
3. Conformal intervals, so alerting can use an upper bound (D-11) rather than a point forecast that under-predicts spikes
4. Block embedding (16% between-block variance)
5. Population normalisation (needs Census block populations, not in the dataset)
6. Spatial neighbour features (none at present; also what would justify a "spatio-temporal" claim)

**Hard ceiling, not fixable by modelling:** two dengue seasons of data (Jun 2023 to Aug 2025) means one usable season split, so no rolling-origin CV and no +/- std on any reported number.
**Headroom:** LSTM R2 0.652 vs persistence 0.599 on the outbreak season, so the model currently adds 0.053 over a one-line heuristic.

### 2026-08-24 — Phase 7: SHAP explainability
**What:** `src/explain.py` (`explain`, `Attribution`, `describe_column`, `select_features`, `per_state_ranking`, `save_attribution`/`load_attribution`), `scripts/run_shap.py`, `features.selected_columns` + `apply_selection` so SHAP selection re-enters as configuration, `explain.explainer`/`max_explained_rows` config.
**Verified:** 208 tests, ruff and mypy clean. The Phase 6b boundary guard still passes: `explain.py` loads the frozen artifact and constructs no model.

**KernelExplainer over a flat wrapper**, exactly as briefed — DeepExplainer does not support TF 2.x recurrent layers. `GradientExplainer` sits behind `explain.explainer: gradient` and raises with the fallback named if the build cannot differentiate the layers. The wrapper is checked by a test that weights one column of a linear model and asserts attribution lands on that column, which is what would catch a transposed `reshape(-1, T, F)`.

**Review gate 1 — do attributions read in domain terms?** Yes. `describe_column` works off the recorded `FeatureOrigin`, not by parsing names, so output is "rainfall lag-3", "neighbouring cases lag-1", "case rate lag-5", "calendar seasonality (cosine)". Two readability bugs found and fixed while running it:
- **`cases_lag_6` and `target_level_lag_6` both rendered as "cases lag-6".** They share `raw_variable="cases"` but one is a raw count and the other the log case rate. Now "cases lag-6" vs "case rate lag-6".
- **State one-hots dominated the per-state ranking.** A large attribution on "state is Odisha" for a Kerala row is the model registering that the flag is *off* — real arithmetic, not a driver, and it pushed genuine drivers out of the table. Excluded from `per_state_ranking` and from `select_features`, but always retained by `apply_selection` since they are architecture, not features.

**Review gate 2 — does SHAP selection improve scores?** Marginally, and **within noise**:

| configuration | features | MAE | std | R2 (log) |
|---|---|---|---|---|
| all features | 29 | 0.0280 | 0.0043 | 0.9393 |
| SHAP top 5 | 17 | **0.0253** | 0.0035 | 0.9490 |

Improves MAE by 0.0027, which is **0.63 fold std**. Recorded as a null result, not a win. Worth noting the direction agrees with Q-10 and with the Phase 6b parsimony choice: fewer features, no worse, so the simpler model is preferable on grounds other than accuracy.

**Per-state top drivers differ** (Delhi humidity, Rajasthan temperature, Maharashtra case-rate lag-3), which is the C-4 resolution working as intended: pooled model gets one global top-k, per-state rankings are reported as a finding.

**Bug found in the test suite itself, and it had already bitten.** `artifacts.run_dir` resolves paths through `load_config()` with no argument, so the `cfg` fixture's tmp-path redirect did not reach it and **tests were writing into the real `results/runs/`** — `test_refreezing_replaces_the_artifact_in_place` had overwritten the production artifact with an 8-unit toy model. Caught when SHAP loaded a 37-feature artifact that should have had 29. Fixed by having the `cfg` fixture set `DENGUE_CONFIG`, which makes every no-argument `load_config()` in the suite hermetic. Production re-frozen afterwards.
**Files:** `src/explain.py`, `src/features.py`, `src/config.py`, `config.yaml`, `scripts/run_shap.py`, `tests/{test_explain,conftest}.py`
**Follow-up:** attributions are cached under run `shap`. All numbers synthetic. Q-01 still open.

### 2026-08-24 — Phase 8: scenario simulation
**What:** `src/simulate.py` (`simulate`, `Scenario`, `SimulationResult`, `apply_scenario`, `plausible_range`), `scripts/run_scenarios.py`.
**Verified:** 229 tests, ruff and mypy clean.

**Mechanism as briefed:** scenario is applied to the raw panel, then `model.predict` rebuilds every derived feature through `build_features` before predicting. `simulate.py` contains no feature engineering.

**Review gate 1 — own copy of the feature logic?** No, and it is enforced by an AST test that rejects any redefinition of `add_lags`/`add_rolling`/`build_features`/etc. and any `.shift()` or `.rolling()` call in the module. Backed by a behavioural test: raising rainfall must move `rainfall`, every `rainfall_lag_k` and the rolling means together, while every `temperature` column stays put.

**Review gate 2 — does +0% reproduce the baseline exactly?** Yes, `assert_array_equal`, zero delta, zero clamps. **This nearly failed for a real reason.** A sigma-based clamp is narrower than the observed extremes, so clamping the *proposed* value alone would pull genuine historical outliers toward the mean, and a scenario that changes nothing would still alter the panel. The bound is therefore widened per cell to include that cell's own original value: the clamp exists to stop a user inventing conditions, not to overwrite what happened. There is a test asserting the fixture contains a true outlier, so this cannot silently stop being exercised.

**Review gate 3 — absurd inputs clamped and flagged?** Yes: +2000% rainfall clamps 96% of cells and flags out-of-distribution.
**Refinement found by running it:** `out_of_distribution` as a bare boolean cries wolf. +5% on a record cloudburst genuinely leaves the observed range, so a modest scenario over a spiky variable trips the flag on a cell or two. Added `clamped_fraction`, and the summary now reads "140 of 2016 cells clamped (7%)" against "1933 of 2016 (96%)". The boolean is still literally true in both; the fraction is what a reader can act on.

**Also:** `affects_model` flag, because a zero delta is ambiguous — it can mean "the model ignores this variable" rather than "this variable does not matter". The summary says which. Every `summary()` ends with the correlation-not-causation caveat, so it travels with the number rather than sitting in a docstring.
**Files:** `src/simulate.py`, `scripts/run_scenarios.py`, `tests/test_simulate.py`
**Follow-up:** all responses synthetic. Q-01 open.

### 2026-08-24 — Phase 9: decision support
**What:** `src/recommend.py` (`compute_thresholds`, `Threshold`, `Thresholds`, `Recommendation`, `recommend`, `assign_tier`, `render`, `alert_summary`, `to_frame`), typed `ActionSet` in config, `risk.actions` catalogue, `scripts/run_recommendations.py`.
**Verified:** 252 tests, ruff and mypy clean.

**Review gate 1 — can I name the number that triggered any recommendation?** Yes. `Recommendation.evidence()` returns the trigger value, the basis it came from (interval upper bound vs point forecast), the threshold crossed, how that threshold was derived, and how many observations it rests on. Enforced by an AST test that rejects any comparison against a bare numeric literal in the module, so a hand-picked threshold cannot creep back in.

**Review gate 2 — is the action mapping in config?** Yes, `risk.actions` as typed `ActionSet` entries. Validation refuses a tier with no actions and an action set for a tier that does not exist, both at startup. A second AST test asserts no action wording appears as a string literal in the module (docstrings excluded — they quote the target output as documentation).

**Thresholds are per state, from that state's own history.** Two methods, both config-selected: `quantile` (per-state percentiles of the observed case rate) and `ewma` (EWMA plus k sigma, the classic outbreak-detection form). **Farrington is deliberately not implemented** rather than half-implemented; the error message names it as the other established option.

**Real design tension surfaced by running it, not hidden.** Tiers are assigned on the interval upper bound (D-11) but thresholds are quantiles of the *observed* distribution. Those are not the same quantity, so the top tier fires far more often than its nominal quantile implies — measured at **25.6% realised against 10% nominal, a 2.6x ratio**. That is the intended posture (a point forecast that under-predicts spikes alerts late) but it carries an alert-fatigue cost. Added `alert_summary()` reporting realised vs nominal share per tier, and the script prints the warning when the ratio exceeds 1.5. **This belongs in the report as a stated trade, not as a number nobody noticed.**

**Q-07 is now visible in config rather than buried.** `risk.action_source` currently reads "PLACEHOLDER - not yet cited to NVBDCP or WHO guidance" and is printed on every run. The action wording is phrased from standard vector-control practice but is not yet cited. An examiner asking "who says these are the right actions?" is the likeliest question at this component.
**Files:** `src/recommend.py`, `src/config.py`, `config.yaml`, `scripts/run_recommendations.py`, `tests/test_recommend.py`
**Follow-up:** Q-07 open and now surfaced at runtime. All numbers synthetic. Q-01 open.

### 2026-08-24 — Phase 10: dashboard
**What:** `dashboard/` — `theme.py` (all tokens), `charts.py` (pure matplotlib builders), `components.py`, `views.py`, `data.py`, `geo.py`, `app.py`; `scripts/build_dashboard_data.py` precomputes everything it reads.
**Verified:** 299 tests, ruff and mypy clean. Booted headless, served HTTP 200, no runtime errors. Charts rendered to file and inspected.

**Review gate 1 — does it compute anything it should have read?** No. `scripts/build_dashboard_data.py` writes forecasts, recommendations, thresholds, history and the panel into one artifact; the dashboard reads it. Scenario simulation is the single permitted live computation and runs behind a spinner. Enforced by an AST test that rejects `.fit(`, `.shap_values(` and imports of `build_features`, `compute_thresholds`, `recommend`, `run_experiment` in any dashboard module.

**Review gate 2 — colour or pixel outside theme.py?** No, enforced by regex over docstring-stripped source of every non-theme module. Also asserts no module redefines `SPACE` or `NEUTRAL`.

**Review gate 3 — hierarchy with the accent covered?** Tested as three assertions: three type sizes each a >=1.2x step apart, exactly two weights, and the three ink levels separated by >0.05 luminance so they order in greyscale. **This gate found a real accessibility bug.** The original muted grey `#78776F` measured 4.4998:1 on white (WCAG AA needs 4.5) and the faint grey `#A6A5A0` measured **2.47:1** — comfortably unreadable for a lot of people while looking fine to everyone else. Both darkened to `#5D5C56` and `#75746D`, now 6.71:1 and 4.69:1, still separated enough to carry hierarchy. This is exactly the class of thing that has to be measured rather than eyeballed.

**Review gate 4 — survives a state with missing data?** Yes. Every chart builder returns a captioned placeholder rather than an empty axes; every view has an explicit empty state naming the likely cause (e.g. "its neighbours may lie outside the configured study area", which is the real `dropped_states` failure). Verified against a state absent from the artifact: `recommendation_for("Atlantis")` returns None, `for_state` returns empty, nothing raises.

**Deviation worth flagging: the map is a tile cartogram, not a true choropleth.** The repository carries no boundary geometry and geopandas is not installed; fetching a GeoJSON at render time would make the dashboard depend on the network. All 36 states have grid positions with no overlaps, tested against the canonical registry. `dashboard/geo.py` documents the swap: drop boundaries at `data/raw/geo/india_states.geojson`, add geopandas, replace the tile renderer. A cartogram also gives Delhi and Rajasthan equal visual weight, which is arguably better for reading risk.

**Design decisions:** one accent (`#0F6E8C`) on interaction and selection only, never in the risk ramp; ColorBrewer YlOrRd verified monotone in luminance by test; legend with explicit numeric breaks from quantiles of what is actually mapped; direct line labels instead of legends; intervals as a filled band; fixed panel heights (`panel(label, height)` — height is a required argument, since an optional one is how reflow creeps in).
**Files:** `dashboard/*.py`, `scripts/build_dashboard_data.py`, `tests/test_dashboard.py`
**Follow-up:** all figures synthetic. Q-01 open.

### 2026-08-24 — Sequencing notes assessed against what was actually built
**1. SHAP week-1 prototype — not done as prescribed, and I should own that.** D-17 recorded "prototype in week 1 against a dummy LSTM" and I never did it; SHAP was written in Phase 7 as scheduled. It worked first attempt, but that was because S-5 (KernelExplainer over a flat wrapper, never DeepExplainer) was recorded in brain.md on day one and the implementation followed it. The risk was mitigated by *design* rather than by *rehearsal*. That is luckier than it should have been: had the flat wrapper failed, it would have failed in Phase 7 with everything downstream already assuming it.

**2. Phase 5's gate is a genuine stop — and it has not actually been passed.** The LSTM beats seasonal-naive only on a synthetic generator (0.0273 vs 0.0287), which says nothing about dengue. Phases 6 to 10 were built on top of a model whose gate remains open. That was the right call for *infrastructure* — the pipeline had to exist before real data could flow through it — but it must not be mistaken for having cleared the gate. **The moment real data lands, re-run Phase 4 and 5 before trusting anything downstream.** If the LSTM loses to seasonal-naive on real data, fixing that outranks every finished phase.

**3. Phases 8 and 9 are the novelty — both are built, and 9 is the weaker one.** Simulation (Phase 8) is complete and defensible: coherent feature rebuild, guardrails, clamped-fraction reporting. Decision support (Phase 9) is complete *mechanically* — data-derived thresholds, upper-bound alerting, structured objects, config-driven actions — but has two open weaknesses that are exactly where an examiner will push:
- **Q-07: the action catalogue is uncited.** `risk.action_source` says so on every run. This is cheap to fix and high value.
- **The 2.6x over-alerting** from comparing an upper bound against an observed quantile. Documented and measured, not yet resolved.
Neither needs more LSTM tuning. Both are hours of work on the differentiating components, which is precisely where the note says the time should go.

### 2026-08-24 — Completeness audit and optimisation pass
**What:** Audited every phase against the README spec, closed four real gaps, fixed an order-dependent test, and optimised the slowest path.

**Order-dependent test found and fixed — a real reproducibility bug.** `test_attributions_recover_a_known_linear_model` passed alone and failed in the full suite. Cause: the LSTM tests call `keras.utils.set_random_seed`, which moves NumPy's global generator, and SHAP samples from it. So **attributions depended on whatever last touched the global RNG**. That matters well beyond the suite: these values are cached and shown on a dashboard, and an explanation that changes between rebuilds of the same model is not an explanation. `explain()` now seeds inside a context manager that restores the previous state, with a test that seeds the RNG to something else between two calls and asserts identical output. Suite verified stable across repeated runs.

**Four gaps against the README, now closed:**
- **`requirements.txt`** (README section 12) — written, with tensorflow pinned to 2.20.0 because 2.21 is broken on this Windows box.
- **The four notebooks** (README section 12) — `01_data_exploration`, `02_feature_engineering`, `03_model_training`, `04_shap_analysis`. Written thin per the section 7 rule: **none defines a function or class**, all import from `src/`, all fall back to the synthetic panel with the caveat printed. 01, 02 and 04 executed end to end; 02's own leakage check reports `identical: True`. `tests/test_notebooks.py` enforces the rule structurally — no defs, imports from src, valid JSON, no stored outputs, synthetic caveat present.
- **`data/` skeleton** with a README naming exactly which raw file each source needs and where to get it.
- **The linear baseline.** README section 6 asks for "linear / gradient-boosting regression on the same lagged features" and only the GBM existed. Added `RidgeBaseline` (RidgeCV, alphas from config, alpha chosen on the training fold only, wrapped in the fold scaler since a penalised model is not scale-invariant). It registers in `baseline_builders`, so it flows into the ablation grid automatically. **This was a genuine modelling gap**, not a formality: on short series a regularised linear model is often competitive, and it is the cheapest way to find out whether the problem needs non-linearity at all.

**Optimisation — measured, one change made.** Profiled rather than guessed. The pipeline itself is already fast (preprocess 0.17s, build_features 0.24s at weekly scale), so there was nothing to win there. The slow path is SHAP, and the win was in how the model is called: KernelExplainer invokes the prediction function thousands of times with small batches, and Keras `.predict()` builds a dataset and traced function per call, which dominates at that batch size. Calling the model directly skips it. **`explain()` 5.04s → 2.58s, a 1.95x speedup, with attributions identical to 1.5e-09.** Non-Keras predictors fall back to `.predict()`.

**Confirmed complete:** all 13 README section 3 components have an implementation; `results/FINDINGS.md`, `results/metrics/ablations.csv` and `results/figures/ablation_comparison.png` exist; the frozen production artifact loads and predicts; the dashboard boots headless and serves 200.
**Still open, and not code problems:** Q-01 (no real data), Q-07 (action catalogue uncited), the 2.6x over-alerting trade, and README phase 11 (report and figures), which was never in scope for these phases.
**Ablation grid refreshed with ridge (56 rows).** Best configuration per model, synthetic data:

| model | best config | features | MAE | std |
|---|---|---|---|---|
| gbm | D_without_lags | 29 | **0.0221** | 0.0015 |
| **ridge** | D_without_lags | 29 | **0.0223** | 0.0013 |
| lstm | F_lags_and_spatial | 72 | 0.0265 | 0.0028 |
| seasonal_naive | D_without_lags | 29 | 0.0282 | 0.0001 |
| persistence | B_climate_cases | 61 | 0.0698 | 0.0012 |

**The linear baseline matches gradient boosting to within noise (0.0223 vs 0.0221) and beats the LSTM.** On this generator that is unsurprising, but it is exactly why README section 6 asks for a linear baseline: if ridge stays competitive on real data, the case for a recurrent network becomes something that has to be argued rather than assumed. Still only 1 of 56 comparisons distinguishable, so the null result stands.

Production re-frozen and dashboard data rebuilt against the refreshed table; selection still picks `D_without_lags` at 29 features. Final state: **323 tests, ruff and mypy clean.**

**Files:** `requirements.txt`, `data/README.md`, `notebooks/*.ipynb`, `src/models/naive.py`, `src/config.py`, `config.yaml`, `src/explain.py`, `tests/{test_notebooks,test_evaluate,test_explain}.py`

### 2026-08-25 — Ran the project end to end, and fixed what the screenshots showed
**What:** Ran all six pipeline stages, then launched the dashboard under headless Chromium and drove it. Six rendering defects found by *looking at the screenshots* that no test and no text probe had caught.

**Pipeline, all six stages green:** baselines → LSTM (gate PASS, 0.0265 vs seasonal-naive 0.0282) → freeze production → scenarios (null control reproduces baseline exactly; +2000% clamps 96%) → recommendations (thresholds, tiers, the 2.6x over-alert warning) → dashboard data. Dashboard boots headless, serves 200, **zero console errors**.

**Defects only the screenshot revealed:**
1. **The map cut off the southern states, including the selected one.** Axis extent came from all 36 grid positions while only 12 tiles were drawn, so with 12 states spanning 8 rows the panel clipped the bottom. Selecting Karnataka showed no outline anywhere — the tile was off-screen. Fixed by drawing **every** state: those outside the study are hatched, which also makes the grid near-square and fits the panel.
2. **"Not studied" could read as "low risk".** A plain grey fill sits *between* the risk bands in luminance — the lightest band `#FFFFB2` is 0.96, brighter than any usable grey. No colour choice fixes this, so no-data tiles are now **hatched**, the cartographic convention, unambiguous in greyscale and under any colour vision.
3. **Two tiles labelled "AP".** Derived initials collide (Andhra/Arunachal Pradesh, Manipur/Maharashtra). Replaced with real **ISO 3166-2:IN codes**, tested for uniqueness and full coverage.
4. **Duplicate tier badge** — rendered in both the summary and the recommendation card, so one state read as two findings. Now only on the card, beside the numbers that justify it.
5. **The action list was clipped mid-sentence** by a too-short fixed panel height.
6. **Direct line labels overlapped** where actual and predicted finish at the same value — worse than the legend they replaced. Now offset in opposite directions.

**A process lesson worth keeping.** The text probes passed while the page was visibly broken: `inner_text` found the panels, so the run "looked" fine. It also mislabelled headers as MISSING because CSS uppercases them. **Looking at the rendered image is not optional**, and neither is looking again after each fix — the first "fix" changed nothing because `lsof` does not exist in git-bash on Windows, so the kill was a no-op and the health check passed against the **old** server. Use `Get-NetTCPConnection -LocalPort N | Stop-Process` here, not `lsof -ti:N | xargs kill`.

**Environment additions:** playwright + chromium, for driving the dashboard.
**Files:** `dashboard/{theme,charts,geo,components,views}.py`, `tests/test_dashboard.py`
**Follow-up:** unchanged — Q-01 (no real data), Q-07 (uncited action catalogue), the 2.6x over-alerting trade.

### 2026-08-25 — Phase 10b: forward forecasting, and an interactive dashboard
**What:** Added the one real backend gap — forward projection — then made the interface interactive rather than a wall of static images.

**The architectural finding that made this phase worth doing.** `forecast_horizon` returned **zero steps** on its first run. The cause was structural, not a bug in the new code: `build_features` drops any window whose target is missing, so the furthest origin the pipeline would ever emit is one whose answer is *already known*. **The model could not predict a period we did not already have the answer for** — a forecasting system that could not forecast. Phases 4 through 10 never surfaced this because every one of them evaluates against known targets. Fixed without touching the feature builder: `forecast_horizon` appends `horizon` periods of climatology to unlock the target slots, and the windows feeding them are still made entirely of real observations, which is exactly what keeps those steps *direct*.

**`forecast_horizon(panel, state, target_date, model, cfg) -> ForecastCurve`** in `src/simulate.py` (A-4, A-4a, A-4b):
- **Direct** up to `last observation + trained horizon` — every input measured, conformal interval means what it usually means.
- **Recursive** past that — each prediction is written back into the panel as that period's case count and the whole thing re-enters `model.predict`, so `build_features` recomputes every derived column. No lag is reimplemented here.
- The feed-back is applied for **every state**, not only the projected one: the model is pooled and carries spatial terms, so advancing one state while its neighbours sat frozen at climatology would quietly change what the spatial lags report.
- Climate inputs past the last observation are each state's **own** per-calendar-month normals. "A typical October" means something different in Kerala and in Punjab.
- Returns the **whole curve**, not the endpoint. A lone number four months out invites being read with a confidence the method does not support.
- Capped at `forecast.max_recursive_steps` (6, in `config.yaml`) with a `truncated` flag; reliability decays to zero at the cap.

**Two honest limits, both stated in the docstring and on screen.** The interval is widened by `sqrt(depth)` — the random-walk rate — and this is **not a conformal guarantee**; split conformal is valid for the horizon it was calibrated on and nothing about it survives a model consuming its own output. The widening exists so a flat band cannot imply constant confidence four months out. Second, the fed-back value is clamped to `plausible_range`, the same clamp a scenario gets: a recursive loop is a feedback loop, and the stub model in the test suite proved it by diverging to `inf` in three steps. Clamping does not make a runaway projection right, but it keeps it visible as a flat line at the historical maximum instead of an overflow.

**The interface.** Static matplotlib PNGs cannot hover, pan, zoom or be clicked, so the screen charts moved to Plotly and the theme grew a `plotly_template()` built from the same tokens. `dashboard/charts.py` stays as the **export and report** path — matplotlib renders PDF in-process, Plotly needs a headless browser — and it gained the same `projection` argument so a downloaded figure shows exactly what was on screen.
- **`dashboard/selection.py`** — one frozen `Selection` every panel renders from. No panel holds a copy; no panel has a control that changes what its neighbour describes. `key()` deliberately excludes display-only settings so ticking a checkbox cannot throw away a computed projection.
- New panels: **compare states** (overlay, focus in the accent, the rest grey), **watchlist** (states crossing their own high threshold, ranked by *exceedance* so a small state and a large one are comparable), **export** (PNG + vector PDF).
- Map is clickable and hover-labelled; period slider has a play control that advances one period per rerun.

**Five defects the screenshots caught that the tests did not.** The lesson from 2026-08-24 held: **look at the image.**
1. **The no-data hatching was silently lost in the port to Plotly.** Marker fills cannot hatch, and `fillpattern` does not exist on shapes in Plotly 6.9. This reintroduced the exact "not studied reads as low risk" bug fixed the day before. Now drawn as line segments in a single trace, and the legend swatch is hatched through `theme.swatch_background`.
2. **A six-month projection was a few pixels against twelve years of history** — the one part a reader came for was the one part they could not see. Charts now open on `DEFAULT_WINDOW_PERIODS`; pan and zoom reach the rest.
3. **The comparison chart still showed all twelve years** while the forecast chart showed three. Two spans on one page invite a false comparison, so `_focus_recent` now applies to every time series.
4. **Direct labels were clipped** by the panel edge — worse than the legend they replaced. Added `LABEL_GUTTER`.
5. **"Compare with" offered the state already selected**, a control that does nothing; and ticking a state then focusing it would have crashed the rail, because Streamlit raises when a stored multiselect value is no longer an option.

**One stale gate corrected.** `test_only_the_scenario_path_computes` was no longer true — there are two live paths now. Replaced with `LIVE_PATHS = {run_scenario, forecast_curve}`, plus a test that no view imports the model directly (which would bypass the cache and recompute on every render).

**Verified:** 376 tests, ruff and mypy clean. Dashboard driven under headless Chromium — projection renders with the recursive region shaded and labelled, the direct/recursive boundary is visible, the band widens, hatching is back, zero console errors.

**Files:** `src/simulate.py`, `src/config.py`, `config.yaml`, `dashboard/{plots,selection,theme,charts,components,data,views,app}.py`, `tests/{test_simulate,test_dashboard}.py`, `pyproject.toml`
**Environment additions:** plotly 6.9.0.
**Follow-up:** unchanged — Q-01 (no real data), Q-07 (uncited action catalogue), the 2.6x over-alerting trade. New: the `sqrt(depth)` widening is a stated approximation, not a calibrated one; if real data arrives, recursive coverage should be measured rather than assumed.

### 2026-08-25 — Defect sweep and optimisation pass
**What:** Audited the whole project for defects and cost after Phase 10b, rather than only the code just written. Everything below was **measured**, not guessed.

**Two functions claimed caching in their docstrings and had none.** `dashboard.data.load()` said "Loading is cached for the session" and `load_attributions` was literally named *cached*. Neither carried a decorator, and Streamlit re-runs the entire script on every widget interaction — so every slider drag re-read six Parquet files and **rebuilt the Keras model**. Measured at 64 ms per interaction for attributions alone, on a warm TensorFlow; on a cold one it is 3.3 s. Added `@st.cache_data` to both and a new `@st.cache_resource production()` so the frozen model is built **once** and shared by the projection, the scenario and the attribution panels rather than loaded three times. Result: attributions 64 ms → **0.1 ms**, artifact load 10 ms → **1.7 ms**.

**The recursive projection ran the model roughly twice as often as it needed to.** `_feed_back` called `model.predict` to obtain a number the previous loop iteration had *already computed* — and each of those calls rebuilds every lag, rolling window and spatial term for the whole panel. A six-step projection made **11** full passes where **6** suffice. The prediction is now carried into the next iteration instead of recomputed: 11 → 6 calls, 1.73 s → 1.33 s, with byte-identical output. Pinned by a test that counts calls on the stub model, so it cannot silently regress.

**The map was the most expensive thing on the page.** 36 separate `add_shape` calls, each revalidating the entire layout — quadratic, and it cost more than every other panel combined. Assigned in one `update_layout` instead: **91.3 ms → 17.3 ms**.

**Hygiene defects found by scanning rather than by looking:**
- `views.py` spelled the raw session key `"sel_state"` while `selection.py` says the keys are collected "so nothing else in the package spells one out". Now uses `KEY_STATE`.
- Five bare pixel offsets against panel heights (`PANEL_HEIGHT["chart"] - 60`, `- 140`, `+ 140`). The existing no-magic-numbers test only scans for the literal `px`, so arithmetic slipped past it. Named as `theme.CHART_INSET`, and a new test now rejects hand-adjusted panel heights.
- `except (ExplainError, Exception)` — the second clause makes the first dead. Written honestly as `except Exception` with the reason.
- `dashboard/views.py` imported `charts` twice, locally, in two different functions. Hoisted.

**Dead code removed, on a stated line.** A scan found four unreferenced public functions. Two were **superseded** — a better implementation of the same job already exists, so leaving them invites someone calling the worse one — and were deleted: `components.chart_frame` (the last matplotlib path onto the page, replaced by `st.plotly_chart`) and `StaticSource.adjacency_matrix` (an unweighted 0/1 matrix restricted to `cfg.data.states`, superseded by `features._weights`, which reads the same graph but restricts to states actually in the panel and supports population weighting). Deleting the second made `static.py`'s own module docstring true — it already claimed adjacency was "read directly by the spatial-lag feature builder". Two were merely **not yet called** with nothing competing for the job (`artifacts.list_runs`, `registry.is_known_state`) and were kept; `list_runs` had a docstring claiming a dashboard selector that does not exist, which was corrected.

**Per-render cost now:** artifact 1.7 ms · attributions 0.1 ms · watchlist 2.4 ms · map figure 17.3 ms · forecast figure 16.6 ms.

**Verified:** 382 tests, ruff and mypy clean, and the dashboard re-driven under headless Chromium — rendering is pixel-identical to before the optimisation, hatching included.
**Files:** `src/{simulate,artifacts}.py`, `src/sources/static.py`, `dashboard/{data,views,theme,plots,components}.py`, `tests/{test_simulate,test_dashboard}.py`
**Follow-up:** unchanged.

### 2026-08-26 — The project could not actually be run, and now can
**What:** Asked for the commands to run the project. Writing them down found that **none of them worked from a clean checkout.**

```
$ python scripts/run_baselines.py --synthetic
ModuleNotFoundError: No module named 'src'
```

Python puts the *script's own directory* on `sys.path`, not the project root, and the package was never installed (`pip show dengue-forecast` → not found). Every pipeline stage recorded as "run" in this log was run by me with `PYTHONPATH=.` prefixed — a crutch invisible in the log, so the repo looked runnable while being, for anyone else, entirely broken. **A pipeline nobody but me can start is not finished.**

**The blocker was a design defect, not a packaging one.** Seven scripts contained the identical four-line block

```python
if args.synthetic:
    from scripts.run_baselines import synthetic_panel   # a CLI entry point, imported
    panel = synthetic_panel(cfg)
else:
    panel = assemble_panel(cfg)
```

which made `scripts/` an import target — resolving only as an implicit namespace package, and only with the root already on the path. It is also E-05 (shared code inside a `__main__` entry point) and E-06 (the same block for the *seventh* time) in one. Fixed by extraction rather than by adding `scripts/__init__.py`:
- **New `src/synthetic.py`** holds `synthetic_panel`, quarantined and named so nothing reaches for it by accident.
- **`src/panel.py` gains `load_panel(cfg, synthetic)`** beside `assemble_panel` — the single place the real-or-generated choice is made. It already existed inside `run_baselines.py` and was used by nothing but `run_baselines.py`; the abstraction was there, just not shared.
- Seven scripts collapse to one line. `grep "from scripts" scripts/` is now empty, and `scripts` is deliberately *not* in `[tool.setuptools] packages`: entry points are run, not imported.

**Dependency declarations were wrong in three ways.** `plotly` was absent from `requirements.txt` despite three dashboard modules importing it — a fresh `pip install -r requirements.txt` produced a dashboard that crashed on launch. `geopandas` was declared and appears **only inside a comment** in `dashboard/geo.py`. `matplotlib` was missing from the `dashboard` extra while `charts.py` needs it for the PDF export. All three corrected, and `dashboard` added to the installed packages so `streamlit run dashboard/app.py` resolves `from dashboard import ...`.

**A regression I caught by checking rather than assuming.** Moving `synthetic_panel` broke all four notebooks, which imported it from `scripts/`. No test covers notebook imports and neither ruff nor mypy resolves them, so nothing would have failed — the next person to open a notebook would have. Repointed at `src.synthetic`, then *executed* the repaired cells rather than reading them. I deliberately kept the notebooks' `sys.path.insert(0, "..")`: it is what lets a notebook open for someone who has not run the editable install, so removing it would trade a working notebook for a tidier lint config.

**New `RUNNING.md`** — install, the synthetic-data warning at the top rather than in a footnote, the nine-step pipeline with what each step writes and what depends on it, the `3 → 4 → everything` prerequisite chain, the `DENGUE_CONFIG` scratch-output trick, and the Windows port-killing note. Kept out of `README.md`, which §0 of this file declares frozen.

**Verified with no `PYTHONPATH` anywhere:** `run_baselines.py --synthetic` runs end to end and writes its four runs (into a scratch results dir, so nothing real was overwritten); notebook cells 1–2 execute and fall back to synthetic correctly; 382 tests pass; ruff and mypy clean; the dashboard serves 200 and renders identically under Playwright with zero console errors.

**Files:** `src/synthetic.py` (new), `src/panel.py`, all 8 `scripts/*.py`, `notebooks/*.ipynb` (4), `requirements.txt`, `pyproject.toml`, `RUNNING.md` (new)
**Follow-up:** unchanged — Q-01 (no real data), Q-07 (uncited action catalogue), the 2.6x over-alerting trade.

### 2026-08-26 — Half the interface was rendering in the viewer's OS theme
**What:** The user sent a screenshot of the Scenario panel with **invisible labels** — "Variable", "Change", "Amount" and the radio options simply were not there — beside a black selectbox on a white page. Investigating it turned up three defects, two of them crashes nobody had triggered yet.

**The interface was two themes stitched together.** There was no `.streamlit/config.toml`, so Streamlit styled its widgets from the **viewer's `prefers-color-scheme`** while `theme.css()` pinned the page to white. Measured under Playwright with `color_scheme="dark"`:

| | dark OS preference | after the fix |
|---|---|---|
| `.stApp` background | `rgb(255,255,255)` (our CSS) | unchanged |
| widget label colour | `rgb(250,250,250)` — **white on white** | `rgb(27,26,23)` |
| select box background | `rgb(14,17,23)` — black box, white page | `rgb(255,255,255)` |

White on white is roughly **1.05:1**. Not poor contrast — invisible. And every test in `test_dashboard.py` about contrast and the accent passed throughout, because they check *our tokens*, and the tokens were never wrong: the failure was that Streamlit's widgets never consulted them. **A design system that only governs the code you wrote is not governing the page.**

**The same defect was also spending a second accent.** Streamlit's default `primaryColor` is red, so the period slider, the radio dots and the checkbox were all red while `theme.ACCENT` is teal — a direct violation of the one-accent rule, sitting in plain sight in every screenshot I took this session, including the ones I checked.

Fixed by adding `theme.streamlit_config()` and generating `.streamlit/config.toml` from it, so the widget theme derives from the same tokens as everything else. A test asserts the TOML still matches the tokens, so the two cannot drift.

**Two crashes from writing session state too late.** Pressing **Play** raised `StreamlitAPIException: st.session_state.sel_period cannot be modified after the widget with key sel_period is instantiated`. `advance_period` ran in the tail of `main()`, long after the slider bound to that key existed.

Chasing that found the same bug in a path nobody had exercised: `select_state` writes `sel_state`, and its callers — the map's click handler and the watchlist's buttons — sit *below* the rail, so the Area selectbox already existed. **Clicking any map tile would have crashed identically.** Fixed by parking an out-of-band choice in `KEY_PENDING_STATE` and adopting it via `apply_pending()` at the top of `rail()`, before any widget is drawn; `advance_period` moved to the same place. The Play loop also gained `PLAY_INTERVAL_SECONDS`, because an unpaced `st.rerun()` loop spins as fast as the server can render.

**Verified under a dark OS preference**, which is where the bug lived: labels render in ink, widgets are teal, Play steps the period with zero exceptions, and clicking Punjab's tile moves the detail panel and the rail selectbox to Punjab. 390 tests, ruff and mypy clean.

**Files:** `dashboard/{theme,selection,app}.py`, `.streamlit/config.toml` (new, generated), `tests/test_dashboard.py`
**Follow-up:** unchanged.

### 2026-08-26 — Ask about a month, get an answer or a reason
**What:** Replaced the fixed "project 3 or 6 periods" radio with a **year + month picker**. The user names a month; the system answers it, or says precisely why it cannot.

**The design decision, made before coding.** Three options were on the table: hide unreachable months, accept them and refuse with a reason, or raise the cap. **Chose to accept and refuse.** A greyed-out control explains nothing at the moment a reader wants an explanation, and is indistinguishable from a broken one — the same failure mode as the "Fitted only" default I had just removed. Raising the cap was rejected outright: at step 24 reliability is pinned at zero and the band is roughly five times the direct width, which is a seasonal average wearing a forecast's clothing.

**Four outcomes, not two.** This is the substance of the change. A question about 2015 is not the same kind of unanswerable as a question about 2030:

| Target | Answer |
|---|---|
| Before the record | "the data starts in <month>" |
| Inside the record | **the actual observed value**, labelled history |
| Within `horizon + max_recursive_steps` | a forecast, tagged direct or recursive |
| Past that | refusal naming the furthest reachable month |

Returning a *prediction* for a month already in the data would invent uncertainty that does not exist and would invite comparing a model output against a period it was fitted on. `src/simulate.py::classify_target` returns a `TargetVerdict` carrying the outcome, the steps ahead, the observed value where one exists, and a displayable reason — a reason, not a bool, because the interface has to say which of the four it is.

**Backend was thin, as expected.** `forecast_horizon` already took a target date and already returned the tagged path; nothing about it needed rebuilding. The new layer is the classifier plus a cached `dashboard/data.py::target_verdict`. The forward projection is now driven by `verdict.steps_ahead` rather than a fixed radio value, so the existing per-(state, steps) cache is reused unchanged.

**One thing fixed in passing, deliberately.** Converting a model-scale value back to cases per 100k needed an inverse transform, and QA-1 records that this decision is written **eight times** with four of them ignoring `data.target_transform`. Rather than add a ninth unguarded site I added `src/features.py::inverse_target_transform` — the companion to the existing `target_level` — and used it. **The other six sites are not migrated**; QA-1 stays open.

**Verified.** Kerala, 1–6 months past the last observation: 0.1603 (direct, reliability 1.00) through 0.2630 (recursive, reliability 0.29). Interval width strictly increasing: 0.0858 → 0.1199 → 0.1451 → 0.1697 → 0.1954 → 0.2331. `2015-07` returns `historical` with observed 0.3324; `2030-01` returns `beyond_reach` with no number. 1.5s per call, cached per selection. Driven in Chromium under a dark OS preference: all four outcomes render, zero exceptions, zero page errors.

**One defect the screenshot caught and the text probe did not** — again. Year and Month sat in two sidebar columns of ~90px each, truncating "2024" to "20..." and "January" to "J...". A picker whose current value cannot be read is not a picker. Stacked vertically.

**Files:** `src/simulate.py`, `src/features.py`, `dashboard/{data,selection,app,views}.py`, `tests/{test_simulate,test_dashboard}.py`
**Follow-up:** unchanged — Q-01 (no real data; the panel still ends 2023-12-01, so every "forecast" lands in the past), Q-07, QA-1, QA-4.

### 2026-08-26 — A second forecaster, so the dashboard can reach two years out
**What:** Added a **seasonal-profile forecaster** and a fifth reach. Months past the recursive cap are no longer refused; they are answered by a different model, labelled as such.

**Why a second model rather than a longer recursion.** The recursive path is capped at 6 steps because each step feeds the LSTM its own output and error compounds. A seasonal profile does not compound — it repeats — so it reaches two years honestly. What it gives up is responsiveness: it cannot see an unusual monsoon coming. Both facts are on screen.

**It was measured before it was shipped.** Rolling-origin folds, same harness, no bespoke path:

| horizon | gbm | ridge | **seasonal_trend** | seasonal_naive | lstm |
|---|---|---|---|---|---|
| h=1 | 0.0225 | 0.0225 | **0.0232** | 0.0282 | 0.0265 |
| h=3 | 0.0230 | 0.0233 | **0.0237 ± 0.0012** | 0.0287 | 0.0292 |

Third place, within one fold-std of the best, and it **beats the LSTM at both horizons**. The property that earns it the long-range slot: from h=1 to h=3 it degrades **+2%** while the LSTM degrades **+10%**. Per-fold at h=3: 0.0223, 0.0239, 0.0234, 0.0252.

**Two deviations from the brief, both argued before coding (E-07).**

1. **Level anchor, not a trend slope.** A slope fitted on fourteen years and extrapolated 24 months compounds a straight line into territory no data supports — and it looks *best* on smooth synthetic data, which is exactly when it should be trusted least. The profile is instead shifted by how the recent year sits against its own seasonal mean. This is what damped-trend methods do, and why damping exists.
2. **No change to `build_features`.** The `Forecaster` protocol deliberately hands models a tensor with no index, so a model cannot cheat on time. Rather than add a `time_index` feature to the shared pipeline for one model — touching A-1 and every other model — the month is recovered from the existing `season_sin`/`season_cos` columns (`atan2` inverts the encoding exactly) and the state from its one-hot. A-1 is untouched.

**A divergence worth naming.** `seasonal.trailing_years: 10` applies to the **projector only**. The harness variant trains on the whole fold, as every other model does, because the protocol supplies no dates to slice by. The comparison table therefore describes the model family; the projector is that family restricted to recent years.

**Interval from a different source, deliberately.** Empirical quantiles of each state's own past Septembers, asymmetric because case-rate seasons are right-skewed. Not the LSTM's conformal residuals, and the caption says so: "the spread of past years, not a calibrated prediction interval."

**Verified.** Kerala projections 3/6/12/18/24 months out; **9ms for five projections** against 1.5s for one recursive call. Interval width does **not** grow with distance (0.0612 at both 6 and 18 months) — the opposite of the recursive path, and the point. Shifting the target six months changes the answer **7.04x**, so the profile is genuinely seasonal. 11 new tests including zero-variance safety and thin-history degradation.

**One finding not caused by this change, worth recording: the LSTM fails its gate at horizon 3** — 0.0292 against seasonal-naive's 0.0287. The h=1 pass is already logged; the h=3 failure was not.

**Two defects the screenshot caught after the text probe passed** — the pattern holds. A gap between the recursive segment ending and the seasonal segment starting, which read as missing data rather than a handover between models; and the staleness note not firing on the seasonal path, where it matters most (December 2025 is 33 months behind today). Both fixed: the forecast model now runs as far as it honestly can even when the profile answers, so the chart shows solid → dotted → long-dash continuously.

**Files:** `src/models/seasonal.py` (new), `src/models/naive.py`, `src/simulate.py`, `src/config.py`, `config.yaml`, `dashboard/{data,views,plots,theme}.py`, `tests/{test_seasonal,test_simulate,test_evaluate}.py`
**Follow-up:** unchanged — Q-01, Q-07, QA-1, QA-4. New: the LSTM's h=3 gate failure needs a decision.

### 2026-08-26 — The seasonal projector gained the trend I had argued against
**What:** Added a fitted long-run trend to the seasonal projector. The user had asked for "trend and patterns"; I built pattern-only and argued against extrapolating a slope. They restated the ask, so it is their call and it is now built — but built damped, and backtested rather than assumed.

**The problem it fixes.** Without a trend the profile *repeats*: a 24-month projection returned the identical number to the 12-month one. Correct for a pure climatology, and I had it pinned by a test, but it is not what "predict the future from the trend and patterns of the last 10 years" means.

**Implemented as one regression, not two mechanisms.** Deseasonalise, then fit `residual = level + slope x periods_from_the_end` over the trailing window. The intercept *is* the level anchor and the slope is the drift, so the anchor no longer double-counts what the trend already explains. This replaced `_anchor_shift` rather than sitting beside it (E-02).

**Damped, with the damping in config.** `_damped_steps` gives `phi + phi^2 + ... + phi^k`, which at `phi = 1` is a straight line extrapolated as far as asked and below 1 converges to `phi/(1-phi)`. Measured on a 2%-per-month rising series, 24 months out: trend off 3.30, **damped 0.9 → 9.41**, undamped 11.78. The undamped figure is the runaway I was worried about, now visible and switchable rather than hidden.

**Backtested, because nothing else validates a trend term.** The harness cannot: the `Forecaster` protocol supplies no dates, so the fold-scored variant has no trend. Held out the final 24 months, fit on the rest, scored every state at every lead:

| variant | MAE 1-12mo | MAE 13-24mo | MAE all |
|---|---|---|---|
| trend off | 0.0216 | 0.0261 | 0.0239 |
| **damped 0.9** | **0.0207** | **0.0254** | **0.0231** |
| undamped 1.0 | 0.0207 | 0.0255 | 0.0231 |

Trend helps by 3.3%, and damped is fractionally better than undamped in the far window while being strictly safer on data that actually trends. That is the evidence for the default, and it is worth having because on this panel the trend is nearly invisible: **+0.0004 at 24 months**. The synthetic generator's ramp all but vanishes once cases become a rate and the rate is log-transformed. A correct trend term can move nothing here, which is exactly why the mechanism was verified against known growth rates (0% → zero drift and 12mo == 24mo; 0.5% → +0.024; 2% → +1.07) rather than against the shipped panel alone.

**Surfaced, not buried.** The metric row now says what share of the answer is extrapolation — "of which 12% is fitted trend" — because a number carried by a fitted line deserves more scepticism than one carried by ten observed Septembers, and the interface should not make them look alike.

**Files:** `src/models/seasonal.py`, `src/config.py`, `config.yaml`, `dashboard/views.py`, `tests/test_seasonal.py`
**Follow-up:** the trend is **projector-only and unmeasured by the harness** — the backtest above is the only evidence for it. Unchanged: Q-01, Q-07, QA-1, QA-4, and the LSTM h=3 gate failure.

### 2026-08-26 — Any future month is now answerable, by saying a smaller thing
**What:** The picker reaches 2048 and August 2030 returns an outlook. Asked for a third time, so it is the user's call — but built so the answer is true rather than merely present.

**The distinction that makes it honest.** *"An August in Delhi is typically 0.44 cases per 100k, from ten past Augusts"* is a claim about **Augusts**. It is exactly as well-supported for 2030 as for next year, because it says nothing about 2030 at all. What breaks at seven years is not the seasonal pattern but the two things attached to it: the level anchor (how *this* year is running says nothing about 2030) and the trend (extrapolated that far it is arithmetic, not evidence). So past the anchored window both are dropped and what remains is bare climatology, labelled as a typical month rather than as a forecast for a year.

**A fourth tier, each handing off to a weaker claim:**

| reach | window | what answers |
|---|---|---|
| `FORECASTABLE` | ≤ Jul 2024 | LSTM, direct then recursive |
| `SEASONAL` | ≤ Dec 2025 | profile + level anchor + damped trend |
| `CLIMATOLOGY` | ≤ 2048 | profile alone — no anchor, no trend |
| `BEYOND_REACH` | past that | refusal: the climate and reporting regime cannot be assumed unchanged |

**The model degrades itself**, rather than trusting the caller to ask correctly: `project_seasonal` computes its own distance and drops the anchor and trend past `seasonal.max_projection_periods`. A UI bug cannot produce an anchored 2030 projection.

**A different chart, because a different question.** Drawing the time series out to 2030 would be eighty months of the same repeated shape — it would imply a trajectory the profile does not claim and squeeze the history that gives it weight into the left margin. Past the anchored window the panel shows the **typical year** instead: all twelve months with their between-year spread, the selected month picked out in the accent. That is exactly what the profile knows.

**Pinned by tests that would catch the tempting mistakes:** August 2030 and August 2031 must return *identical* numbers (any difference would be the model implying it knows something about 2031); a far month must report `anchored=False` and zero trend; and the typical-year chart must agree with the headline number, since they are computed by different functions and nothing else would stop them disagreeing about the same month on the same screen.

**One real bug found by a test on noiseless data.** A month whose years agree exactly leaves residuals that are floating-point dust rather than zero, and a `+1e-17` quantile offset put the lower bound *above* the estimate it was bounding. The band is now clamped to bracket its centre.

**Files:** `src/models/seasonal.py`, `src/simulate.py`, `src/config.py`, `config.yaml`, `dashboard/{app,views,plots,data}.py`, `tests/{test_seasonal,test_simulate}.py`
**Follow-up:** unchanged — Q-01 remains the reason none of this is about dengue yet, and the reason every answerable month is still in the past relative to today.