"""Freeze the winning configuration as the one artifact everything downstream loads.

Thin CLI entry point. Selection and fitting live in :mod:`src.production`.

Re-running this replaces the artifact in place, which is the point: a new month of
data means running this one step, and every downstream phase picks up the new model
without changing a line.

Usage::

    python scripts/freeze_production.py --synthetic
    python scripts/freeze_production.py --synthetic --experiment F_lags_and_spatial
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd

from src.artifacts import run_dir
from src.config import load_config
from src.panel import load_panel
from src.preprocess import preprocess
from src.production import PRODUCTION_RUN, select_configuration, train_production

warnings.filterwarnings("ignore")


def main(argv: list[str] | None = None) -> int:
    """Select the winning configuration, refit it, and write the artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", action="store_true",
                        help="use a generated panel instead of data/ (not real results)")
    parser.add_argument("--experiment", default=None,
                        help="override the configuration chosen from the ablation table")
    parser.add_argument("--horizon", type=int, default=None)
    args = parser.parse_args(argv)

    cfg = load_config()

    panel = load_panel(cfg, args.synthetic)
    clean = preprocess(panel, cfg).panel[list(panel.columns)]

    experiment, horizon = args.experiment, args.horizon
    if experiment is None or horizon is None:
        table_path = Path(cfg.paths.metrics) / "ablations.csv"
        if not table_path.is_file():
            raise SystemExit(
                f"no ablation table at {table_path}. Run scripts/run_ablations.py "
                "first, or pass --experiment and --horizon explicitly."
            )
        chosen, chosen_horizon = select_configuration(pd.read_csv(table_path), cfg)
        experiment = experiment or chosen
        horizon = horizon if horizon is not None else chosen_horizon
        print(f"selected from ablation table: {experiment} at horizon {horizon}")

    artifact = train_production(clean, cfg, experiment=experiment, horizon=horizon)

    print(f"\nfrozen: {run_dir(PRODUCTION_RUN)}")
    print(f"  model       : {cfg.production.model}")
    print(f"  experiment  : {artifact.experiment}")
    print(f"  horizon     : {artifact.spec.horizon}")
    print(f"  features    : {artifact.spec.n_features}")
    print(f"  calibration : {len(artifact.residuals)} residuals")
    print(f"  trained at  : {artifact.trained_at.isoformat()}")

    forecasts = artifact.predict(clean)
    print(f"\nend-to-end check: {len(forecasts)} forecasts from the raw panel")
    print(forecasts.tail(3).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    if args.synthetic:
        print("\nSYNTHETIC DATA - this artifact is a rehearsal, not a deliverable.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
