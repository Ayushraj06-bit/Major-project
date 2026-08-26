"""Multi-source spatio-temporal dengue outbreak prediction.

Module map:

* :mod:`src.config`    — frozen, validated settings loaded once from ``config.yaml``
* :mod:`src.io`        — the single Parquet caching decorator
* :mod:`src.artifacts` — run storage that the dashboard reads from
* :mod:`src.sources`   — ``DataSource`` protocol and per-feed implementations
* :mod:`src.features`  — ``build_features``: raw panel to model-ready tensors
* :mod:`src.splits`    — rolling-origin cross-validation
* :mod:`src.models`    — ``Forecaster`` protocol, baselines, GBM, LSTM, conformal wrapper
* :mod:`src.evaluate`  — ``run_experiment``: metrics and artifacts for one configuration
* :mod:`src.explain`   — SHAP attribution over the fitted model
* :mod:`src.simulate`  — what-if scenarios, re-running the same feature pipeline
* :mod:`src.recommend` — risk tiering and decision support
"""

__version__ = "0.1.0"
