"""Produce decision-support recommendations from the frozen model.

Thin CLI entry point. Loads the production artifact and the cached SHAP
attributions; trains nothing and recomputes nothing.

Usage::

    python scripts/run_recommendations.py --synthetic
"""

from __future__ import annotations

import argparse
import sys
import warnings

import pandas as pd

from src.config import load_config
from src.explain import load_attribution
from src.features import build_features
from src.panel import load_panel
from src.preprocess import preprocess
from src.production import load_production
from src.recommend import (
    alert_summary,
    compute_thresholds,
    recommend,
    render,
    to_frame,
)

warnings.filterwarnings("ignore")


def _drivers(spec, top_k: int) -> dict:
    """Cached SHAP drivers keyed by (state, origin_date), or empty if none cached."""
    from src.explain import ExplainError

    try:
        attribution = load_attribution()
    except ExplainError:
        print("no cached SHAP attributions; recommendations will omit drivers\n")
        return {}

    return {
        (state, pd.Timestamp(date)): attribution.top_drivers(spec, row=position, k=top_k)
        for position, (state, date) in enumerate(attribution.sample_index)
    }


def main(argv: list[str] | None = None) -> int:
    """Derive thresholds, produce recommendations, and print them."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", action="store_true",
                        help="use a generated panel instead of data/ (not real results)")
    parser.add_argument("--limit", type=int, default=8, help="how many to print")
    args = parser.parse_args(argv)

    cfg = load_config()
    model = load_production()

    panel = load_panel(cfg, args.synthetic)
    clean = preprocess(panel, cfg).panel[list(panel.columns)]

    thresholds = compute_thresholds(clean, cfg)
    print(f"thresholds derived by: {thresholds.method}")
    print(thresholds.frame().head(6).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print()

    forecasts = model.predict(clean)
    _, _, spec = build_features(clean, model.cfg, horizon=model.spec.horizon)
    recommendations = recommend(forecasts, thresholds, cfg, drivers=_drivers(spec, top_k=2))

    summary = alert_summary(recommendations, cfg)
    print("tier distribution (realised vs what the quantiles imply):")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    high = summary.iloc[-1]
    if high["ratio"] > 1.5:
        print()
        print(
            f"NOTE: {high['tier']} fires {high['ratio']:.1f}x more often than its "
            f"{high['nominal_share']:.0%} quantile implies. That is structural, not a "
            "bug: tiers are assigned on the interval upper bound while thresholds "
            "come from the observed distribution. Deliberate (brain.md D-11), but it "
            "is the alert-fatigue trade and belongs in the report."
        )
    print()

    alerts = [item for item in recommendations if item.is_alert]
    ranked = sorted(alerts, key=lambda item: -item.trigger_value_log)[: args.limit]
    print(f"=== TOP {len(ranked)} ALERTS (of {len(alerts)}) ===")
    for item in ranked:
        print(f"\n[{item.state}, target {item.target_date.date()}]")
        print(render(item))

    print("\n=== MACHINE-READABLE (first 3) ===")
    print(to_frame(recommendations[:3]).to_string(index=False))

    print(f"\naction catalogue source: {cfg.risk.action_source}")
    if args.synthetic:
        print("\nSYNTHETIC DATA - these recommendations are a rehearsal, not advice.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
