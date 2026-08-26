"""Run what-if scenarios against the frozen production model.

Thin CLI entry point. Loads the artifact; never trains, never rebuilds features
itself.

Usage::

    python scripts/run_scenarios.py --synthetic
"""

from __future__ import annotations

import argparse
import sys
import warnings

from src.config import load_config
from src.panel import load_panel
from src.preprocess import preprocess
from src.production import load_production
from src.simulate import Scenario, simulate

warnings.filterwarnings("ignore")

#: A small standing set, so runs are comparable between sessions.
SCENARIOS = (
    Scenario(variable="rainfall", change=0.0, label="null control (+0%)"),
    Scenario(variable="rainfall", change=20.0, label="+20% rainfall"),
    Scenario(variable="rainfall", change=-20.0, label="-20% rainfall"),
    Scenario(variable="temperature", change=2.0, mode="absolute", label="+2 degrees"),
    Scenario(variable="rainfall", change=2000.0, label="absurd: +2000% rainfall"),
)


def main(argv: list[str] | None = None) -> int:
    """Run the standing scenarios and print each result with its caveats."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", action="store_true",
                        help="use a generated panel instead of data/ (not real results)")
    args = parser.parse_args(argv)

    cfg = load_config()
    model = load_production()
    print(f"frozen model: {model.experiment} h{model.spec.horizon} "
          f"({model.spec.n_features} features)\n")

    panel = load_panel(cfg, args.synthetic)
    clean = preprocess(panel, cfg).panel[list(panel.columns)]

    for scenario in SCENARIOS:
        result = simulate(clean, scenario, model, cfg)
        print("=" * 72)
        print(result.summary())
        print()

    if args.synthetic:
        print("SYNTHETIC DATA - these responses describe the generator, not dengue.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
