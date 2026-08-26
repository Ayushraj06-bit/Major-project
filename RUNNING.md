# Running the project

How to install and run this repository. [README.md](README.md) is the frozen
specification; this file is the operational companion to it.

---

## ⚠️ Read this first: the numbers are synthetic

**No real dengue data has been obtained yet.** `data/raw/` is empty, so every
command below is run with `--synthetic`, which generates a seasonal panel of the
shape the pipeline expects.

Every figure the pipeline produces this way **describes a sine-wave generator, not
dengue.** They demonstrate that the machinery is correct end to end; they are not
findings, and they must not be reported as any. Runs built this way are named with
a `synthetic_` prefix in `results/runs/` so they cannot be confused with real ones,
and the dashboard prints the same warning in its sidebar.

When real data arrives, drop `--synthetic` from every command. Nothing else changes.

---

## Setup

Python 3.11 (3.11.9 is what this is verified against).

```bash
pip install -r requirements.txt
pip install -e .
```

**Both steps are required.** The second one is not optional packaging etiquette:
the scripts live in `scripts/` and import from `src/`, but Python puts the
*script's own directory* on `sys.path`, not the project root. Without the editable
install every entry point fails immediately:

```
$ python scripts/run_baselines.py --synthetic
ModuleNotFoundError: No module named 'src'
```

`pip install -e .` puts `src` and `dashboard` on the path properly, once, for
every shell. The alternative — prefixing all nine commands with `PYTHONPATH=.` —
has different syntax in PowerShell and git-bash and is easy to forget.

TensorFlow is pinned to 2.20.0 on purpose: 2.21 is broken on Windows, where the
tflite DLL fails to import.

---

## The short version

Everything below has already been run and its results are on disk. To just open
the dashboard:

```bash
streamlit run dashboard/app.py
```

---

## The full pipeline

Run from the project root, in this order. Steps 3 and 4 are a hard prerequisite
chain; the rest are independent of each other but all need step 4.

| # | Command | Writes | Needed by |
|---|---------|--------|-----------|
| 1 | `python scripts/run_baselines.py --synthetic` | `results/runs/synthetic_*` | the comparison table |
| 2 | `python scripts/run_lstm.py --synthetic` | LSTM run | the Phase 5 gate |
| 3 | `python scripts/run_ablations.py --synthetic` | `results/metrics/ablations.csv` | **step 4** |
| 4 | `python scripts/freeze_production.py --synthetic` | `results/runs/production` | **steps 5–8** |
| 5 | `python scripts/run_shap.py --synthetic` | `results/runs/shap` | the Attribution panel |
| 6 | `python scripts/run_scenarios.py --synthetic` | scenario runs | the write-up |
| 7 | `python scripts/run_recommendations.py --synthetic` | thresholds and tiers | the write-up |
| 8 | `python scripts/build_dashboard_data.py --synthetic` | `results/runs/dashboard` | **the dashboard** |
| 9 | `streamlit run dashboard/app.py` | — | — |

Notes on the order:

* **Step 3 → 4.** `freeze_production.py` reads `results/metrics/ablations.csv` to
  choose which configuration to freeze. Without it, it exits with a message
  telling you to run the ablations first, or to pass `--experiment` and
  `--horizon` explicitly.
* **Step 4 → everything after.** The frozen artifact in `results/runs/production`
  is the single boundary the later phases load. Re-freezing is the one step you
  repeat when new data arrives; the phases downstream pick up the new model
  without changing a line.
* **Step 8 → 9.** The dashboard is a thin reader. It does no training and no SHAP;
  it reads what step 8 precomputed. If it reports a missing artifact, run step 8.
* **Step 5** is optional in the sense that the dashboard degrades gracefully
  without it — the Attribution panel says so rather than breaking.

Useful flags: every script takes `--synthetic`; `run_lstm.py`, `run_baselines.py`
and `freeze_production.py` take `--horizon`; `freeze_production.py` takes
`--experiment` to override the ablation table's choice.

---

## Using the dashboard

The left rail drives every panel at once — area, period, and how far ahead to
project. Two controls worth knowing:

* **Forward projection.** Past the model's trained horizon, the projection becomes
  *recursive*: the model reads its own earlier predictions, with climatological
  normals standing in for weather nobody has observed. Those steps are drawn
  dotted, the region is shaded, and the interval widens with each one. That
  widening is an honest acknowledgement, **not** a coverage guarantee — split
  conformal is only valid for the horizon it was calibrated on.
* **Export.** PNG and vector PDF, drawn by the same matplotlib code as the report
  figures, so what you download is what the write-up will contain.

---

## Development

```bash
pytest                      # 382 tests
ruff check .                # lint
mypy dashboard/ src/        # type check
```

Tests redirect every path into a temporary directory, so running them cannot
overwrite anything in `results/`.

To run the pipeline against a scratch output directory instead of the real one,
point `DENGUE_CONFIG` at a copy of `config.yaml` with `paths.results` and
`paths.runs` changed:

```bash
DENGUE_CONFIG=/path/to/scratch/config.yaml python scripts/run_baselines.py --synthetic
```

### Windows note

To free the port between dashboard restarts, use PowerShell:

```powershell
Get-NetTCPConnection -LocalPort 8501 | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }
```

`lsof -ti:8501 | xargs kill` does **not** work here — `lsof` does not exist in
git-bash on Windows, so the kill silently does nothing and the next health check
passes against the old server.
