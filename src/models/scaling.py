"""Feature scaling, as a wrapper around any :class:`~src.models.Forecaster`.

Scaling belongs strictly inside the cross-validation fold, fitted on training rows
alone. The question is *where* to put it, and there are two options that both
satisfy that rule:

1. ``run_experiment`` scales before calling the model.
2. A wrapper that is itself a ``Forecaster`` scales as part of its own ``fit``.

This module takes option 2, because option 1 breaks the naive baselines. A
persistence forecaster predicts by reading the currently-observed value straight
out of its input; if ``run_experiment`` has already standardised that column, the
value it reads is a z-score, not a case rate, and its prediction is meaningless.
Making ``run_experiment`` scale for some models and not others would require it to
branch on model type — exactly what the ``Forecaster`` protocol exists to avoid.

As a wrapper, scaling is composed per model instead::

    lstm_factory = lambda: StandardScaled(LSTM(cfg))     # needs scaling
    gbm_factory  = lambda: GBMBaseline(cfg)              # trees do not care
    naive        = lambda: Persistence(column)           # must see raw values

The fold guarantee is unchanged and in fact stronger: the wrapper's ``fit`` is
only ever handed one fold's training rows, so it is structurally incapable of
seeing the test period.
"""

from __future__ import annotations

import numpy as np

from src.models import Forecaster


class StandardScaled:
    """Standardise features per column, then delegate to a wrapped forecaster.

    Statistics are computed across samples *and* timesteps for each feature, which
    is the right pooling: a feature means the same thing at every position in the
    window, so it should be scaled by one set of numbers rather than by a different
    pair per timestep.

    Constant features — a static attribute, or a column that happens not to vary in
    one fold — have zero variance. Their scale is replaced by 1.0 rather than
    dividing by zero, leaving them centred at zero and contributing nothing, which
    is the honest representation of a feature that carries no information here.
    """

    def __init__(self, base: Forecaster) -> None:
        self.base = base
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> StandardScaled:
        """Fit the scaler on the training rows only, then fit the wrapped model.

        The validation block is transformed with the *training* statistics, never
        refitted on. Refitting would let the validation distribution inform the
        transform, which is the same leak in a smaller place.
        """
        flat = X.reshape(-1, X.shape[-1])
        self.mean_ = flat.mean(axis=0)
        scale = flat.std(axis=0)
        self.scale_ = np.where(scale > 0.0, scale, 1.0)

        scaled_validation = None
        if validation is not None:
            X_val, y_val = validation
            scaled_validation = (self.transform(X_val), y_val)

        self.base.fit(self.transform(X), y, scaled_validation)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Apply the fitted scaling, then delegate."""
        return self.base.predict(self.transform(X))

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Standardise using the statistics fitted in :meth:`fit`."""
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("StandardScaled.transform called before fit")
        return (X - self.mean_) / self.scale_

    @property
    def diagnostics(self) -> dict[str, float]:
        """Whatever the wrapped model reports, so wrapping hides nothing."""
        inner = getattr(self.base, "diagnostics", None)
        return inner if isinstance(inner, dict) else {}

    @property
    def statistics(self) -> dict[str, list[float]]:
        """Fitted per-feature mean and scale.

        Recorded per fold in the run result, so that "the scaler was fitted inside
        the fold" is a checkable claim rather than an assertion in a docstring —
        the numbers must differ from one fold to the next.
        """
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("StandardScaled.statistics read before fit")
        return {"mean": self.mean_.tolist(), "scale": self.scale_.tolist()}
