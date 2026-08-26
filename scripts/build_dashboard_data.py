"""Precompute everything the dashboard reads.

The dashboard is a thin reader. This script does the work once and stores it, so
that opening the interface costs a Parquet read rather than a model load, a SHAP
pass and a threshold derivation.

Re-run it whenever the production model is re-frozen.

Usage::

    python scripts/build_dashboard_data.py --synthetic
"""

from __future__ import annotations

import argparse
import sys
import warnings

import numpy as np
import pandas as pd

from src.artifacts import save_run
from src.config import load_config
from src.features import target_level
from src.panel import load_panel
from src.preprocess import preprocess
from src.production import load_production
from src.recommend import compute_thresholds, recommend, to_frame

warnings.filterwarnings("ignore")

DASHBOARD_RUN = "dashboard"


def observed_history(panel: pd.DataFrame, cfg) -> pd.DataFrame:
    """Observed case rate per state and date, on the interpretable scale."""
    levels = target_level(panel, cfg)
    return (
        pd.DataFrame({"actual": np.expm1(levels)})
        .reset_index()
        .rename(columns={"date": "date"})
        .sort_values(["state", "date"])
    )


def main(argv: list[str] | None = None) -> int:
    """Compute forecasts, thresholds, recommendations and history, and store them."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", action="store_true",
                        help="use a generated panel instead of data/ (not real results)")
    args = parser.parse_args(argv)

    cfg = load_config()
    model = load_production()

    panel = load_panel(cfg, args.synthetic)
    clean = preprocess(panel, cfg).panel[list(panel.columns)]

    print("forecasting...")
    forecasts = model.predict(clean)

    print("deriving thresholds...")
    thresholds = compute_thresholds(clean, cfg)

    print("building recommendations...")
    recommendations = recommend(forecasts, thresholds, cfg)

    save_run(
        DASHBOARD_RUN,
        overwrite=True,
        forecasts=forecasts,
        recommendations=to_frame(recommendations),
        thresholds=thresholds.frame(),
        history=observed_history(clean, cfg),
        panel=clean.reset_index(),
        meta={
            "experiment": model.experiment,
            "horizon": model.spec.horizon,
            "n_features": model.spec.n_features,
            "trained_at": model.trained_at.isoformat(),
            "interval_coverage": 1.0 - cfg.conformal.alpha,
            "threshold_method": thresholds.method,
            "action_source": cfg.risk.action_source,
            "synthetic": bool(args.synthetic),
        },
    )

    print(f"\nstored under run {DASHBOARD_RUN!r}:")
    print(f"  forecasts       {len(forecasts):>6}")
    print(f"  recommendations {len(recommendations):>6}")
    print(f"  states          {forecasts['state'].nunique():>6}")
    if args.synthetic:
        print("\nSYNTHETIC DATA - the dashboard will render a rehearsal, not results.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
