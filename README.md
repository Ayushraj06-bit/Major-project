# Multi-Source Spatio-Temporal Dengue Outbreak Prediction using Explainable Deep Learning

An AI-based dengue outbreak intelligence and decision-support framework for Indian states.

The system does not stop at forecasting. It follows a five-stage workflow:

> **Predict → Explain → Simulate → Recommend → Visualize**

---

## 1. Overview

Conventional dengue forecasting projects stop at "how many cases next week?" — a question that has been answered many times and offers limited research depth. This project extends that baseline in four directions:

1. **Multi-source fusion** — climatic, epidemiological, demographic and public-awareness signals combined into a single input representation, instead of climate alone.
2. **Temporal lag modelling** — explicit encoding of the delay between environmental conditions and case counts.
3. **Explainability** — SHAP attribution so every prediction can be traced to the features that drove it.
4. **Actionability** — scenario simulation and a decision-support layer that turn a number into a recommended course of action.

### Research questions

| # | Question | Addressed by |
|---|----------|--------------|
| RQ1 | Does combining heterogeneous data sources improve forecasting over single-source input? | Ablation study (§6) |
| RQ2 | Which temporal lags and which data sources contribute most to predictions? | Lag ablation + SHAP |
| RQ3 | Can XAI meaningfully explain a deep temporal model's behaviour on epidemiological data? | SHAP module |
| RQ4 | How sensitive is predicted risk to changes in environmental conditions? | Scenario simulation |

---

## 2. System Architecture

```
Multi-Source Data Collection
    │
    ├── Climatic        (rainfall, temperature, humidity)
    ├── Epidemiological (historical dengue cases per state)
    ├── Demographic     (population density, population)
    └── Public awareness (Google Trends search interest)
    │
    ▼
Preprocessing  ──►  Lag Creation & Data Fusion  ──►  LSTM Forecasting
                                                          │
                                                          ▼
                                                 Dengue Risk Prediction
                                                          │
        ┌──────────────────┬──────────────────┬───────────┴────────────┐
        ▼                  ▼                  ▼                        ▼
  Explainable AI    Scenario Simulation   Recommendation /      Geospatial Heatmap
     (SHAP)            (what-if)          Decision Support          Dashboard
```

---

## 3. Scope — Component Status

| # | Component | Status |
|---|-----------|--------|
| 1 | Multi-state dengue forecasting (India) | Core |
| 2 | Multi-source data integration | Core |
| 3 | Temporal lag features | Core — research contribution |
| 4 | Data fusion into unified input | Core |
| 5 | LSTM forecasting model | Core — primary model |
| 6 | Ablation experiments (MAE / RMSE / R²) | Core — research evaluation |
| 7 | SHAP explainability | Core |
| 8 | Scenario simulation (what-if) | Core |
| 9 | Recommendation / decision support | Core — **design not finalised** |
| 10 | Interactive geospatial heatmap dashboard | Core |
| 11 | Naive baselines (persistence, seasonal-naive) | Recommended |
| 12 | Spatial lag features (neighbouring states) | Recommended |
| 13 | Prediction intervals (MC dropout) | Recommended |
| 14 | Random Forest baseline | Optional |

---

## 4. Data Sources

| Category | Variables | Candidate source | Granularity |
|----------|-----------|------------------|-------------|
| Epidemiological | Dengue case counts | NVBDCP / IDSP | State, monthly or weekly |
| Climatic | Rainfall, temperature, humidity | IMD, ERA5 reanalysis | Daily, gridded → aggregated to state |
| Demographic | Population, population density | Census of India | Static per state |
| Public awareness | Search interest ("dengue", "dengue symptoms", "dengue fever") | Google Trends | Weekly, state level |

### Critical note on data availability

**This is the single largest project risk and must be resolved before any modelling begins.** Indian dengue case data is most reliably published at **state level, monthly or annually**. If weekly district-level data cannot be obtained, the following change:

- Sequence length and lag windows become monthly, not weekly.
- Available time points may drop to ~100–150 per state, which strongly constrains LSTM capacity.
- The dashboard becomes a state-level choropleth rather than a district heatmap.

Locate and verify the actual data files first. Every downstream design decision depends on it.

### Note on "social media"

The public-awareness signal is **Google Trends search interest**, not social-media scraping. Trends is free, structured, and API-accessible; X/Twitter data now requires paid access and would constitute a project of its own. Search interest also carries a caveat: it may *lag* outbreaks (people search after news coverage). If it only contributes at lag 0 or negative lags, it is a nowcasting signal rather than a predictive one — a legitimate finding to report either way.

---

## 5. Methodology

### 5.1 Feature engineering

**Temporal lags.** Climatic conditions do not affect case counts immediately — rainfall creates breeding sites, which raise the mosquito population, which raises transmission, typically over 2–6 weeks. Lagged copies of each driver are therefore created:

```
rainfall_lag_1 … rainfall_lag_k
temperature_lag_1 … temperature_lag_k
humidity_lag_1 … humidity_lag_k
cases_lag_1 … cases_lag_k
trends_lag_1 … trends_lag_k
```

Rolling aggregates (e.g. 4-week mean rainfall) may be added alongside point lags.

**Spatial lags.** To justify the spatio-temporal framing, each state also receives neighbouring states' case counts at t-1, t-2, weighted by adjacency. Dengue spreads through human movement, so a neighbour's outbreak last period is genuine signal — and this yields an additional ablation ("with vs. without spatial features").

**Static features.** Population density does not vary week to week and teaches an LSTM little as a sequence input. Use it either as a static feature concatenated after the recurrent layers, or as a normaliser by predicting **cases per 100,000** — which also makes states directly comparable.

### 5.2 Prediction target

Must be fixed explicitly before training:

- **Target variable:** `log(cases + 1)` is recommended for count data with sharp spikes — it stabilises variance and prevents a single outbreak period from dominating the loss.
- **Forecast horizon:** 4 periods ahead is far more useful for public health than 1-step-ahead (which approaches trivial autocorrelation) and gives real actionable lead time.
- **Granularity:** one shared model with state as a feature, or one model per state.

### 5.3 Model

LSTM is the primary and required forecasting model, selected because this is fundamentally a sequential forecasting problem. Random Forest was dropped from the core pipeline — maintaining two full pipelines adds training time, preprocessing complexity and debugging effort without strengthening the research direction. It may be retained as an optional baseline if time allows.

**Training configuration (starting point):**

| Parameter | Value |
|-----------|-------|
| Max epochs | 100 |
| Early stopping | patience 10 |
| Batch size | 32 |
| Optimizer | Adam |
| Loss | MSE |

### 5.4 Validation

- **Time-based splitting only.** Random shuffling leaks future information into training and invalidates results.
- **Rolling-origin cross-validation.** Train 1–100 → test 101–110; train 1–110 → test 111–120; and so on. Report mean ± std across folds. With ~150 time points a single split is close to a coin flip; rolling origin is the difference between a result and an anecdote.
- **Scale inside each fold.** Fit the scaler on the training portion only and apply it to validation/test. Scaling the whole series up front is a classic and easily-spotted leakage bug.

---

## 6. Experiments

The research evaluation compares **data configurations**, not competing algorithms.

**Data-source ablation:**

| Configuration | Inputs |
|---------------|--------|
| A | Climate only |
| B | Climate + historical cases |
| C | Full multi-source (climate + cases + demographic + Trends) |

**Feature-design ablation:**

| Configuration | Description |
|---------------|-------------|
| D | Without lag features |
| E | With lag features |
| F | With lag + spatial lag features |

**Baselines** — every configuration must be compared against:

1. **Persistence** — carry the last observed value forward.
2. **Seasonal naive** — the value from the same period last year.
3. **Linear / gradient-boosting regression** on the same lagged features.

If the LSTM cannot beat seasonal naive, that is a finding worth discovering in month 2, not month 6. These baselines also restore comparative rigour after Random Forest was dropped.

**Metrics:** MAE, RMSE, R².

---

## 7. Explainable AI (SHAP)

The XAI module attributes each prediction to the contributing features — rainfall, humidity, temperature, historical cases, population density, search interest — and their specific lags.

**Practical warnings:**

- SHAP on sequence models is slower and more fragile than on tabular data. `DeepExplainer` frequently breaks on Keras/TensorFlow version mismatches. Fallbacks are `GradientExplainer`, or aggregating over the time axis and using `KernelExplainer` with a small background sample.
- **Prototype this in week 1 against a dummy LSTM.** Do not let it surprise you in the final month.
- Decide the presentation format early. Per-feature values summed over timesteps are far more readable than a raw `timesteps × features` attribution grid.

---

## 8. Scenario Simulation

Users modify environmental inputs — for example "+20% rainfall" or "+2 °C" — and observe how predicted risk shifts relative to the baseline forecast.

**Implementation requirement.** The simulator must operate as:

```
modify raw series → re-run full feature engineering → predict
```

**not** as a direct edit of the final feature vector. If a user raises rainfall, then `rainfall_lag_1`, `rainfall_lag_2`, rolling means and every other derived feature must move together. Changing only `rainfall_t` produces an internally contradictory input the model never saw in training, and its response to it is meaningless.

**Guardrails.** Clamp simulated inputs to a plausible range (historical min/max, or ±2σ) and warn the user when they exit it. Extrapolation beyond the training distribution is precisely where neural networks produce confident nonsense.

**Interpretation caveat.** The model learns correlation, not causation. "Risk rises when rainfall rises" is a statement about the model's learned behaviour, not a causal claim about dengue transmission. Stating this explicitly in the report is a strength — it converts a limitation into a considered discussion point.

---

## 9. Recommendation / Decision Support

**Status: design not yet finalised.** This is the newest component and the least specified — and because it originated from guide feedback, it is the part most likely to be probed in review.

It must not be an arbitrary if-else table. The intended approach is a **risk-tiered, threshold-driven policy** where thresholds are derived from data (quantiles of the historical case distribution, or an established outbreak-detection method such as EWMA or the Farrington algorithm) rather than chosen by hand.

Target output format:

> **Risk: HIGH** — predicted 88 cases (80% interval upper bound 140), above the 90th historical percentile of 62.
> **Top drivers (SHAP):** rainfall lag-3, humidity lag-2.
> **Recommended actions:** source-reduction drive in affected districts; hospital bed pre-positioning; targeted public-awareness messaging.

Every element traces back to a number, which makes the recommendation defensible under questioning.

**Prediction intervals.** Use Monte Carlo dropout — keep dropout active at inference, run ~100 forward passes, take percentiles. Roughly ten lines of code, and it lets alerts trigger on the upper bound rather than the mean, which is far more appropriate for public-health preparedness.

---

## 10. Geospatial Heatmap Dashboard

An interactive dashboard displaying state-wise dengue risk on a map, with risk categorised into low / medium / high bands.

Intended panels:

- Choropleth map of predicted risk by state
- Temporal trend charts (actual vs. predicted)
- SHAP explanation panel for the selected state
- Scenario simulation controls with live re-prediction
- Recommendation panel for the selected state and risk tier

---

## 11. Suggested Tech Stack

| Layer | Tools |
|-------|-------|
| Data handling | pandas, numpy |
| Modelling | TensorFlow / Keras (LSTM), scikit-learn (baselines, scaling) |
| Explainability | shap |
| Geospatial | geopandas, folium or plotly |
| Dashboard | Streamlit or Dash |
| Visualization | matplotlib, seaborn, plotly |
| Data acquisition | pytrends (Google Trends), cdsapi (ERA5) |

---

## 12. Suggested Repository Structure

```
dengue-prediction/
├── data/
│   ├── raw/                 # untouched source downloads
│   ├── interim/             # cleaned, aligned to common time index
│   └── processed/           # fused, lagged, model-ready
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_shap_analysis.ipynb
├── src/
│   ├── data/                # loaders, cleaners, fusion
│   ├── features/            # lag creation, spatial lags, scaling
│   ├── models/              # LSTM, baselines, training loop
│   ├── explain/             # SHAP wrappers
│   ├── simulate/            # scenario engine
│   └── recommend/           # decision-support logic
├── dashboard/
│   └── app.py
├── results/
│   ├── figures/
│   └── metrics/
├── requirements.txt
└── README.md
```

---

## 13. Development Roadmap

| Phase | Work | Gate |
|-------|------|------|
| 1 | Locate and verify all data sources; confirm granularity | **Blocking** — do this first |
| 2 | Preprocessing, alignment to a common time index, fusion | Clean unified dataset exists |
| 3 | Lag + spatial feature engineering | Features validated, no leakage |
| 4 | Naive baselines + metrics harness | Baseline scores recorded |
| 5 | LSTM training, rolling-origin validation | LSTM beats seasonal naive |
| 6 | Ablation experiments (A–F) | RQ1, RQ2 answered |
| 7 | SHAP integration (prototype early, in parallel with Phase 5) | Attributions render correctly |
| 8 | Scenario simulation engine | Lags move coherently |
| 9 | Recommendation logic | Thresholds data-derived |
| 10 | Dashboard assembly | End-to-end demo |
| 11 | Report, figures, documentation | Submission |

---

## 14. Known Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Weekly/district case data unavailable | Reshapes the entire pipeline | Resolve in Phase 1 before any modelling |
| Too few time points for an LSTM | Overfitting; LSTM loses to naive baselines | Small network, heavy regularisation, pooled multi-state model, honest baseline reporting |
| SHAP incompatible with the LSTM build | XAI module fails late | Prototype in week 1; keep GradientExplainer as fallback |
| Google Trends lags rather than leads outbreaks | Feature adds no predictive value | Test explicitly; report a null result as a finding |
| Recommendation layer looks arbitrary | Weakest point under review | Derive thresholds from data, cite an established outbreak-detection method |
| Title claims "spatio-temporal" but model is temporal only | Overclaiming | Add spatial lag features (§5.1) |

---

## 15. Planned Improvements

Drawn from current literature, principally Chen & Moraga (2025), *Forecasting dengue across Brazil with LSTM neural networks and SHAP-driven lagged climate and spatial effects*, BMC Public Health 25:973 — the closest published work to this project, with open code at `github.com/ChenXiang1998/LSTM-Based-Dengue-Prediction-Across-Brazil`.

### Modelling

| # | Improvement | Why |
|---|-------------|-----|
| 1 | **SHAP for feature selection**, not only explanation — train an initial LSTM on all candidate variables, keep the top ~5 by mean absolute SHAP, per state | Cuts redundancy among correlated climate variables; makes XAI a pipeline component instead of a final decoration; yields a second finding (which drivers matter where) |
| 2 | **Pooled multi-state model** with state embedding, target as cases per 100,000 | ~60 monthly points per state is far too few for a per-state LSTM; pooling gives ~2,000 rows |
| 3 | **Cyclic seasonality features** — `sin(2πw/52)`, `cos(2πw/52)` | Captures annual patterns climate alone misses (behaviour, vector-control cycles, reporting) |
| 4 | **Count-appropriate loss** — `log(cases+1)` with MSE, or Poisson / negative-binomial | Dengue counts are overdispersed; raw-count MSE lets one outbreak month dominate training |
| 5 | **Smaller network** — 32–64 LSTM units, dropout, early stopping | The reference paper uses 1,000 units on 364 weeks × 27 states; that capacity would memorise a smaller Indian dataset |
| 6 | **Forecast at 1 and 3 months**, not 1 step ahead | Short-horizon forecasts give no lead time for public health action |

### Evaluation

| # | Improvement | Why |
|---|-------------|-----|
| 7 | **Conformal prediction intervals** — rolling-window residual quantiles around the point forecast | Distribution-free, cheaper than MC dropout, adapts as transmission changes; lets alerts trigger on the upper bound |
| 8 | **Add CRPS** alongside MAE / RMSE / R² | Scores the calibration of intervals, not just point accuracy |
| 9 | **Report negative ablation results honestly** | In Brazil, spatial features *hurt* in sparsely-connected northern states; expect the same in India's northeast — knowing when a method fails is a finding |

### Data

| # | Improvement | Why |
|---|-------------|-----|
| 10 | **Population-weighted climate aggregation** from gridded ERA5 to state level | A flat spatial mean over a large state is dominated by empty area, not where people live |
| 11 | **Replace pytrends** — archived April 2025 and broken on first call; use `trendspy`, manual Trends CSV export, or the Wikipedia Pageviews API | Pipeline will not run otherwise; Pageviews is an official, stable API measuring the same public-attention construct |

### Implementation

| # | Improvement | Why |
|---|-------------|-----|
| 12 | **Wrap the LSTM for SHAP** — flatten input, reshape inside a predict function, use `KernelExplainer` | `DeepExplainer` does not support TF 2.x recurrent layers; this is version-proof, if slow |
| 13 | **Prototype SHAP in week 1** against a dummy LSTM | Longest-standing known failure point in this stack |

### Positioning

Because the Brazil paper already covers LSTM + SHAP + lagged climate + neighbour effects, the defensible novelty of this project is:

1. First application of the framework to **India** and its monsoon-driven seasonality
2. **Scenario simulation** — absent from this literature
3. **Decision support** — absent from this literature
4. **Public-awareness signals** — the reference work uses climate and spatial data only

Frame the report as adapting an established framework to India and **extending it from forecasting to decision support**. This is more defensible than claiming architectural novelty, which examiners can disprove with one search.

---

## 16. Project Identity

**Title (fixed):** *Multi-Source Spatio-Temporal Dengue Outbreak Prediction using Explainable Deep Learning*

The research contribution is centred on multi-source data fusion, temporal lag analysis, explainable AI, scenario simulation, decision support and geospatial visualization — not on algorithm-versus-algorithm comparison.

The system is designed to answer four questions:

1. **What is likely to happen?** — LSTM forecasting
2. **Why does the model predict it?** — SHAP explainability
3. **What could happen if conditions change?** — scenario simulation
4. **What actions could be considered?** — decision support
