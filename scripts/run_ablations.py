"""Run the whole ablation grid, write the results table, and plot the comparison.

Thin CLI entry point. The loop lives in :mod:`src.experiments`.

Usage::

    python scripts/run_ablations.py --synthetic
    python scripts/run_ablations.py --synthetic --models persistence seasonal_naive gbm
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib

from src.config import load_config
from src.experiments import REPORTED_METRICS, run_ablations, significance
from src.panel import load_panel
from src.preprocess import preprocess

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore")


def comparison_plot(table, path: Path, metric: str = "mae_cases_per_100k") -> Path:
    """Grouped bars of every configuration, with fold standard deviation as error bars.

    The error bars are the point of the figure. If they overlap, the ranking is not
    a result, and a reader should be able to see that without reading the table.
    """
    horizons = sorted(table["horizon"].unique())
    fig, axes = plt.subplots(
        1, len(horizons), figsize=(7.5 * len(horizons), 5), squeeze=False, sharey=True
    )

    experiments = list(dict.fromkeys(table["experiment"]))
    models = sorted(table["model"].unique())
    width = 0.8 / max(len(models), 1)

    for column, horizon in enumerate(horizons):
        axis = axes[0][column]
        subset = table[table["horizon"] == horizon]
        for offset, model in enumerate(models):
            rows = subset[subset["model"] == model].set_index("experiment")
            values = [rows[metric].get(name, float("nan")) for name in experiments]
            errors = [rows[f"{metric}_std"].get(name, float("nan")) for name in experiments]
            positions = [i + offset * width for i in range(len(experiments))]
            axis.bar(positions, values, width, yerr=errors, capsize=3, label=model)

        axis.set_xticks([i + 0.4 - width / 2 for i in range(len(experiments))])
        axis.set_xticklabels(experiments, rotation=30, ha="right", fontsize=8)
        axis.set_title(f"horizon {horizon}")
        axis.grid(axis="y", alpha=0.3)
        if column == 0:
            axis.set_ylabel(metric.replace("_", " "))
    axes[0][-1].legend(fontsize=8)

    fig.suptitle("Ablation comparison (mean +/- std across folds)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> int:
    """Run the grid and write table, plot and significance summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", action="store_true",
                        help="use a generated panel instead of data/ (results are not real)")
    parser.add_argument("--models", nargs="*", default=None,
                        help="restrict to these model names")
    args = parser.parse_args(argv)

    cfg = load_config()
    panel = load_panel(cfg, args.synthetic)

    cleaned = preprocess(panel, cfg)
    clean_panel = cleaned.panel[list(panel.columns)]

    table, skipped = run_ablations(
        clean_panel,
        cfg,
        models=tuple(args.models) if args.models else None,
        progress=lambda label: print(f"  running {label}", flush=True),
    )

    metrics_dir = Path(cfg.paths.metrics)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    table_path = metrics_dir / "ablations.csv"
    table.to_csv(table_path, index=False)

    plot_path = comparison_plot(table, Path(cfg.paths.figures) / "ablation_comparison.png")

    print("\n=== ABLATION RESULTS (mean across folds) ===")
    shown = ["experiment", "model", "horizon", "n_features", *REPORTED_METRICS[:4], "coverage"]
    print(table[shown].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    if skipped:
        print("\n=== SKIPPED ===")
        for item in skipped:
            print(f"  {item.experiment} / {item.model} / h{item.horizon}: "
                  f"{item.reason.splitlines()[0][:100]}")

    print("\n=== IS ANY DIFFERENCE DISTINGUISHABLE? ===")
    gaps = significance(table)
    print(gaps.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    decisive = gaps[gaps["distinguishable"]]
    print(f"\n{len(decisive)} of {len(gaps)} comparisons exceed one fold standard deviation.")
    if decisive.empty:
        print("Everything is within noise. Report a null result, not a ranking.")

    print(f"\ntable: {table_path}\nplot : {plot_path}")
    if args.synthetic:
        print("\nSYNTHETIC DATA - these results describe the generator, not dengue.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
