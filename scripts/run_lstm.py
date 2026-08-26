"""Run the LSTM alongside every baseline, on one shared set of folds.

Thin CLI entry point. The fold loop, scaling and metrics all live in
:mod:`src.evaluate`; nothing here trains anything itself.

Usage::

    python scripts/run_lstm.py --synthetic
"""

from __future__ import annotations

import argparse
import sys
import warnings

from src.config import load_config
from src.evaluate import compare, run_experiment
from src.features import build_features
from src.models.lstm import pooled_lstm
from src.models.naive import baseline_factories
from src.panel import load_panel
from src.preprocess import preprocess

warnings.filterwarnings("ignore")


def main(argv: list[str] | None = None) -> int:
    """Run baselines plus the LSTM and print the comparison and the gate verdict."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", action="store_true",
                        help="use a generated panel instead of data/ (results are not real)")
    parser.add_argument("--horizon", type=int, default=None)
    args = parser.parse_args(argv)

    cfg = load_config()
    prefix = "synthetic_" if args.synthetic else ""

    panel = load_panel(cfg, args.synthetic)

    cleaned = preprocess(panel, cfg)
    data = build_features(cleaned.panel[list(panel.columns)], cfg, horizon=args.horizon)
    X, _, spec = data
    print(f"X={X.shape}  sequence={len(spec.sequence_columns)} "
          f"static={len(spec.static_columns)} state={len(spec.state_columns)}")
    print(f"target: {spec.target_name}\n")

    results = [
        run_experiment(factory, data, cfg, f"{prefix}{name}", save=True)
        for name, factory in baseline_factories(spec, cfg).items()
    ]
    results.append(
        run_experiment(pooled_lstm(spec, cfg), data, cfg, f"{prefix}lstm", save=True)
    )

    table = compare(results)
    print(table[["mae_cases_per_100k", "rmse_cases_per_100k", "r2_log", "mae_log"]].to_string())

    lstm = results[-1]
    print("\nLSTM per-fold training diagnostics:")
    for fold in lstm.folds:
        d = fold.diagnostics
        print(f"  fold {fold.number}: epochs {d.get('epochs_run', 0):.0f} "
              f"(best {d.get('best_epoch', 0):.0f})  "
              f"train {d.get('final_train_loss', float('nan')):.4f}  "
              f"val {d.get('final_val_loss', float('nan')):.4f}  "
              f"gap {d.get('train_val_gap', float('nan')):+.4f}")

    seasonal = next(r for r in results if r.name.endswith("seasonal_naive"))
    print(f"\nGATE  LSTM MAE {lstm.primary:.4f}  vs  seasonal-naive {seasonal.primary:.4f}")
    print("      PASS: LSTM beats seasonal naive" if lstm.primary < seasonal.primary
          else "      FAIL: LSTM loses to seasonal naive. Stop and diagnose.")
    if args.synthetic:
        print("\nSYNTHETIC DATA - this verdict describes the generator, not dengue.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
