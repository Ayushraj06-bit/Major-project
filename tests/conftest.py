"""Shared fixtures: a temporary config and a synthetic panel.

Tests never touch the real ``config.yaml`` paths, so nothing they run writes into
``data/`` or ``results/``.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.config import (
    CONFIG_PATH_ENV_VAR,
    Config,
    clear_config_cache,
    load_config,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEST_STATES = ("Kerala", "Odisha", "Tamil Nadu")
TEST_VARIABLES = ("cases", "rainfall", "temperature")


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    """A real config with paths redirected into a temporary directory."""
    raw = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text(encoding="utf-8"))
    raw["data"]["states"] = list(TEST_STATES)
    raw["data"]["start_date"] = "2015-01-01"
    raw["data"]["end_date"] = "2017-12-31"
    raw["split"]["initial_train_size"] = 12
    raw["split"]["test_size"] = 6

    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    # Point the env var at the fixture too, not just this call. Modules that ask
    # for the config with no argument -- artifacts.run_dir among them -- would
    # otherwise resolve the real config.yaml and write into the project's own
    # results/runs, letting a test overwrite the production artifact.
    previous = os.environ.get(CONFIG_PATH_ENV_VAR)
    os.environ[CONFIG_PATH_ENV_VAR] = str(path)
    clear_config_cache()

    yield load_config(path)

    if previous is None:
        os.environ.pop(CONFIG_PATH_ENV_VAR, None)
    else:
        os.environ[CONFIG_PATH_ENV_VAR] = previous
    clear_config_cache()


@pytest.fixture
def panel(cfg: Config) -> pd.DataFrame:
    """A complete synthetic panel with a known seasonal shape and no gaps."""
    from src.panel import complete_index

    index = complete_index(cfg)
    month = index.get_level_values("date").month.to_numpy()
    rng = np.random.default_rng(cfg.project.seed)

    seasonal = np.sin(2 * np.pi * month / 12)
    return pd.DataFrame(
        {
            "cases": 50 + 30 * seasonal + rng.normal(0, 2, len(index)),
            "rainfall": 100 + 80 * seasonal + rng.normal(0, 5, len(index)),
            "temperature": 27 + 4 * seasonal + rng.normal(0, 0.5, len(index)),
        },
        index=index,
    )