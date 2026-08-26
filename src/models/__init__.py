"""Forecasting models.

One module per model (naive baselines, GBM, LSTM, the conformal wrapper), each
satisfying :class:`Forecaster`. Implementations arrive in Phases 4-5 — this module
defines only the contract.

A single protocol across persistence, seasonal-naive, gradient boosting and the
LSTM is what lets the whole ablation run as one loop instead of a branch per
model.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Forecaster(Protocol):
    """A model that learns from a feature tensor and predicts the transformed target.

    Contract notes that bind every implementation:

    * **Input shape.** ``X`` is 3-D ``(n_samples, timesteps, n_features)``. Tabular
      models flatten it inside their own ``fit``/``predict`` rather than asking
      callers for a different shape, so one bundle feeds every model. The SHAP
      wrapper already needs that flatten/reshape step, so the logic is shared.
    * **No scaling inside.** Implementations neither fit nor apply scalers.
      Scaling is fitted per fold by ``run_experiment``, which is what makes
      train/test leakage structurally impossible instead of a rule to remember.
    * **Target scale.** ``y`` is the transformed target — ``log(cases_per_100k + 1)``
      under the default config. Inverting the transform for reporting is the
      evaluation layer's job, not the model's.
    * **Fresh instances.** ``fit`` may assume it is called once on an unfitted
      object. ``run_experiment`` therefore takes a :data:`ForecasterFactory`, not
      an instance; reusing one instance across folds would silently continue
      training on the previous fold's weights.
    """

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> Forecaster:
        """Fit on one training fold.

        Args:
            X: ``(n_samples, timesteps, n_features)`` training tensor.
            y: ``(n_samples,)`` transformed target.
            validation: Optional ``(X_val, y_val)`` from the fold's validation
                block, for early stopping. Supplied to every model uniformly;
                models that do not need it ignore it.

                It is passed in rather than carved from the tail of ``X`` because
                ``X`` is ordered state-major, not time-major. Its tail is one
                state, not one time period, so an internal carve would validate on
                a held-out *state* while claiming to validate on held-out *time*.
                :mod:`src.splits` already cuts a proper time block; this hands it
                over.

        Returns:
            ``self``, so fitting can be chained.
        """
        ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the transformed target for each sample.

        Args:
            X: ``(n_samples, timesteps, n_features)`` tensor, shaped as in ``fit``.

        Returns:
            ``(n_samples,)`` predictions on the same scale as the ``y`` passed to
            ``fit``.
        """
        ...


#: How ``run_experiment`` obtains a model. A factory, never a built instance, so
#: every cross-validation fold trains from scratch.
ForecasterFactory = Callable[[], Forecaster]
