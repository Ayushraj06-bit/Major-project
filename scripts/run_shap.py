"""Explain the frozen production model, cache the attributions, and test the selection.

Thin CLI entry point. Loads the production artifact; never trains anything.

Usage::

    python scripts/run_shap.py --synthetic
    python scripts/run_shap.py --synthetic --evaluate-selection
"""

from __future__ import annotations

import argparse
import sys
import warnings

import pandas as pd

from src.config import load_config
from src.explain import (
    explain,
    per_state_ranking,
    save_attribution,
    select_features,
)
from src.features import build_features
from src.panel import load_panel
from src.preprocess import preprocess
from src.production import load_production

warnings.filterwarnings("ignore")


def main(argv: list[str] | None = None) -> int:
    """Compute attributions, report them in domain terms, and cache them."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", action="store_true",
                        help="use a generated panel instead of data/ (not real results)")
    parser.add_argument("--evaluate-selection", action="store_true",
                        help="rerun the harness with only the SHAP-selected columns")
    args = parser.parse_args(argv)

    cfg = load_config()
    artifact = load_production()
    print(f"explaining frozen model: {artifact.experiment} h{artifact.spec.horizon} "
          f"({artifact.spec.n_features} features)\n")

    panel = load_panel(cfg, args.synthetic)
    clean = preprocess(panel, cfg).panel[list(panel.columns)]

    X, y, spec = build_features(clean, artifact.cfg, horizon=artifact.spec.horizon)
    scaled = (X - artifact.scaler_mean) / artifact.scaler_scale

    print(f"running {cfg.explain.explainer} explainer on "
          f"{min(cfg.explain.max_explained_rows, len(X))} of {len(X)} rows...")
    attribution = explain(artifact.predictor, scaled, spec, cfg)
    save_attribution(attribution, spec)

    print("\n=== TOP FEATURES (mean absolute SHAP) ===")
    importance = attribution.global_importance(spec)
    print(importance.head(12)[["readable", "mean_abs_shap"]].to_string(
        index=False, float_format=lambda v: f"{v:.5f}"))

    print("\n=== BY RAW DRIVER ===")
    print(attribution.by_raw_variable(spec).to_string(
        index=False, float_format=lambda v: f"{v:.5f}"))

    print("\n=== ONE PREDICTION EXPLAINED ===")
    state, date = attribution.sample_index[0]
    print(f"{state}, forecast origin {pd.Timestamp(date).date()}")
    for label, value in attribution.top_drivers(spec, row=0, k=5):
        print(f"  {label:<40} {value:+.5f}")

    print("\n=== PER-STATE RANKING (top driver per state) ===")
    ranking = per_state_ranking(attribution, spec)
    for state in ranking.columns:
        top = ranking[state].idxmax()
        print(f"  {state:<18} {top[1]}")

    selected = select_features(attribution, spec, cfg)
    print(f"\n=== SHAP-SELECTED TOP {cfg.explain.top_k_features} ===")
    for column in selected:
        print(f"  {column}")
    print("\nTo use these, set features.selected_columns in config.yaml.")

    if args.evaluate_selection:
        _evaluate(clean, artifact, selected, cfg)

    if args.synthetic:
        print("\nSYNTHETIC DATA - these attributions describe the generator, not dengue.")
    return 0


def _evaluate(clean: pd.DataFrame, artifact, selected: tuple[str, ...], cfg) -> None:
    """Review gate: does SHAP-based selection actually improve scores?"""
    import dataclasses

    from src.evaluate import run_experiment
    from src.models.lstm import pooled_lstm

    print("\n=== DOES SELECTION HELP? ===")
    rows = []
    for label, columns in (("all features", ()), (f"top {len(selected)}", selected)):
        variant = dataclasses.replace(
            artifact.cfg,
            features=dataclasses.replace(artifact.cfg.features, selected_columns=columns),
        )
        data = build_features(clean, variant, horizon=artifact.spec.horizon)
        result = run_experiment(
            pooled_lstm(data[2], variant), data, variant,
            f"shap_selection_{len(columns) or 'all'}", save=True,
        )
        rows.append({
            "configuration": label,
            "n_features": data[2].n_features,
            "mae_cases_per_100k": result.mean["mae_cases_per_100k"],
            "std": result.std["mae_cases_per_100k"],
            "r2_log": result.mean["r2_log"],
        })
    table = pd.DataFrame(rows)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    gap = table["mae_cases_per_100k"].iloc[1] - table["mae_cases_per_100k"].iloc[0]
    spread = max(table["std"].max(), 1e-12)
    verdict = "improves" if gap < 0 else "does not improve"
    print(f"\nSelection {verdict} MAE by {abs(gap):.4f} ({abs(gap) / spread:.2f} fold std).")
    if abs(gap) / spread < 1.0:
        print("Within noise. Record it as a null result, not a win.")


if __name__ == "__main__":
    sys.exit(main())
