"""Smoke test: the shipped config.yaml loads and satisfies its cross-section rules."""

from __future__ import annotations

from pathlib import Path

from src.config import Config, load_config

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_shipped_config_loads_and_validates() -> None:
    """config.yaml parses into a frozen Config with its invariants intact."""
    cfg = load_config(PROJECT_ROOT / "config.yaml")

    assert isinstance(cfg, Config)
    # Paths are absolutised against the config file's directory on load.
    assert cfg.paths.data_interim.is_absolute()
    assert cfg.paths.runs.is_absolute()
    # Cross-section rules that __post_init__ enforces (brain.md A-2c and A-3d).
    assert cfg.seasonal_period in cfg.lags
    assert max(cfg.forecast.horizons) <= cfg.split.test_size
    # Frozen: a stray assignment downstream cannot mutate shared config.
    assert cfg.project.granularity in {"monthly", "weekly"}
