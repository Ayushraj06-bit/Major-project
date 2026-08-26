"""A generated stand-in for data that has not been obtained yet.

**Nothing here is dengue.** Every number this module produces describes a sine
wave with noise added to it. It exists so the pipeline can be exercised end to
end — features, folds, training, conformal calibration, SHAP, the dashboard —
while the real case data is still absent (brain.md Q-01).

Quarantined in its own module, and named so that no caller reaches for it by
accident. Runs built on it are prefixed ``synthetic_`` in ``results/runs/`` for
the same reason: a generated result must never be mistakable for a finding.

The single caller is :func:`src.panel.load_panel`, which is where the choice
between real and generated data is made explicitly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import Config
from src.panel import complete_index


def synthetic_panel(cfg: Config) -> pd.DataFrame:
    """A seasonal panel with the columns the default config expects.

    Deliberately simple: an annual sine per state, a slow trend, and noise. A
    persistence baseline should do reasonably on it and seasonal-naive very well,
    which is a useful sanity check on the harness itself.
    """
    index = complete_index(cfg)
    month = index.get_level_values("date").month.to_numpy()
    rng = np.random.default_rng(cfg.project.seed)
    seasonal = np.sin(2 * np.pi * (month - 6) / 12)
    trend = np.arange(len(index)) % 168 * 0.02

    return pd.DataFrame(
        {
            "cases": np.abs(80 + 60 * seasonal + trend + rng.normal(0, 8, len(index))),
            "rainfall": np.abs(120 + 100 * seasonal + rng.normal(0, 15, len(index))),
            "temperature": 27 + 5 * seasonal + rng.normal(0, 1.0, len(index)),
            "humidity": np.clip(70 + 15 * seasonal + rng.normal(0, 3, len(index)), 0, 100),
            "search_interest": np.abs(45 + 25 * seasonal + rng.normal(0, 5, len(index))),
            "population": 3.0e7,
            "population_density": 850.0,
        },
        index=index,
    )
