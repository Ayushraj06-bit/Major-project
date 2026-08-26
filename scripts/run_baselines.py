"""Run every baseline and record the scores in ``results/runs/``.

Thin CLI entry point: all the work lives in :mod:`src.evaluate` and
:mod:`src.models.naive`.

Usage::

    python scripts/run_baselines.py                 # real panel from data/
    python scripts/run_baselines.py --synthetic     # generated stand-in

``--synthetic`` exists because the real case data has not been obtained yet. It
generates a seasonal panel of the shape the pipeline expects, so the harness can
be exercised end to end. **Numbers produced this way describe the generator, not
dengue**, and every run is named with a ``synthetic_`` prefix so they cannot be
mistaken for results.
"""

from __future__ import annotations

import argparse
import sys

from src.config import load_config
from src.evaluate import compare, run_experiment
from src.features import build_features
from src.models.naive import baseline_factories
from src.panel import load_panel, summarise_panel
from src.preprocess import preprocess


def main(argv: list[str] | None = None) -> int:
    """Run the baselines and print the comparison table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="use a generated panel instead of data/ (results are not real)",
    )
    parser.add_argument(
        "--horizon", type=int, default=None, help="forecast lead time (default: first configured)"
    )
    args = parser.parse_args(argv)

    cfg = load_config()
    prefix = "synthetic_" if args.synthetic else ""

    panel = load_panel(cfg, args.synthetic)
    print(summarise_panel(panel, cfg).describe())
    print()

    cleaned = preprocess(panel, cfg)
    print(cleaned.describe())
    print()

    data = build_features(cleaned.panel[list(panel.columns)], cfg, horizon=args.horizon)
    X, _, spec = data
    print(f"X={X.shape}  features={spec.n_features}  horizon={spec.horizon}")
    print(f"target: {spec.target_name}")
    print()

    results = [
        run_experiment(factory, data, cfg, f"{prefix}{name}", save=True)
        for name, factory in baseline_factories(spec, cfg).items()
    ]
    for result in results:
        print(result.summary())
        print()

    table = compare(results)
    print("comparison (mean across folds, sorted by MAE on cases per 100k):")
    print(table[["mae_cases_per_100k", "rmse_cases_per_100k", "r2_log", "mae_log"]].to_string())

    best = table.index[0]
    print(f"\nbar to beat: {best} at MAE {table.loc[best, 'mae_cases_per_100k']:.4f} cases/100k")
    if args.synthetic:
        print("\nSYNTHETIC DATA - these numbers describe the generator, not dengue.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
