"""The pooled LSTM forecaster.

Sized for the data that actually exists, not for the data the reference paper had.
Chen and Moraga use 1,000 units on 364 weeks across 27 Brazilian states; the same
capacity on roughly 60 monthly points per Indian state would memorise the training
set and learn nothing transferable. This is 32 to 64 units in one or two layers,
with dropout, and it stops on validation loss.

**Pooling is what makes it trainable at all.** Per-state models are not an option
at this series length. One model sees every state, and state identity enters as a
learned embedding concatenated *after* the recurrence, so the network can shift
its level per state without spending recurrent capacity on an identity that never
changes within a window. Predicting cases per 100,000 rather than raw counts is
the other half of that: it puts Kerala and Uttar Pradesh on a comparable scale.

Static features join at the same point, for the same reason. Repeating population
density across twelve timesteps teaches an LSTM nothing except that it is constant.

There is no training loop here. ``fit`` builds the network and calls Keras; the
fold loop, the scaling and the metrics all live in :mod:`src.evaluate`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.config import Config
from src.features import FeatureSpec
from src.models import Forecaster, ForecasterFactory


class LSTMError(RuntimeError):
    """Raised when the network cannot be built or fitted as configured."""


#: Cached so repeated calls do not re-register the same classes with Keras.
_LAYER_CACHE: tuple[Any, Any] | None = None


def register_serializable_layers() -> tuple[Any, Any]:
    """Define the two column-routing layers, registered for serialisation.

    **Must be called before loading a saved model.** Registration happens when this
    runs, not at import, so a process that only loads an artifact has to call it
    first or Keras cannot resolve ``ColumnSelect``.

    Deliberately not ``keras.layers.Lambda``. A Lambda closes over whatever its
    body references, which here would be the ``PooledLSTM`` instance, and Keras
    then cannot serialise the model at all. Registered layers with a ``get_config``
    make the saved ``.keras`` file standalone, which is what the production
    artifact needs: one file that reloads without the training code being
    reconstructed around it.

    Defined inside a function so importing this module does not require Keras to
    be installed.
    """
    global _LAYER_CACHE
    if _LAYER_CACHE is not None:
        return _LAYER_CACHE

    import keras
    from keras import layers, ops

    @keras.saving.register_keras_serializable(package="dengue")
    class ColumnSelect(layers.Layer):  # type: ignore[misc]
        """Take a fixed set of column indices along one axis."""

        def __init__(self, indices: list[int], axis: int, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.indices = [int(index) for index in indices]
            self.axis = int(axis)

        def call(self, inputs: Any) -> Any:
            """Gather the configured indices."""
            return ops.take(inputs, self.indices, axis=self.axis)

        def compute_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
            """Replace the selected axis with the number of indices kept."""
            shape = list(input_shape)
            shape[self.axis] = len(self.indices)
            return tuple(shape)

        def get_config(self) -> dict[str, Any]:
            """Everything needed to rebuild this layer from the saved file."""
            return {**super().get_config(), "indices": self.indices, "axis": self.axis}

    @keras.saving.register_keras_serializable(package="dengue")
    class LastTimestep(layers.Layer):  # type: ignore[misc]
        """Take the final timestep of a sequence, the forecast origin."""

        def call(self, inputs: Any) -> Any:
            """Slice the last position on the time axis."""
            return inputs[:, -1, :]

        def compute_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
            """Drop the time axis."""
            return (input_shape[0], input_shape[2])

    _LAYER_CACHE = (ColumnSelect, LastTimestep)
    return _LAYER_CACHE


class PooledLSTM:
    """One LSTM across every state, with state identity as a learned embedding.

    Column routing comes from the :class:`~src.features.FeatureSpec`, so which
    columns are sequential, static or state-identity is derived from how each was
    built rather than from hardcoded names.

    Args:
        spec: Feature spec describing the arrays this model will be fitted on.
        cfg: Loaded configuration; ``model.lstm`` supplies every hyperparameter.
        mc_dropout: Keep dropout active at prediction time. Off by default. Phase 6
            uses conformal intervals rather than MC dropout, but the switch is here
            because a dropout ensemble is a cheap second opinion on the conformal
            width, and retrofitting it later would mean touching a trained model.
    """

    def __init__(self, spec: FeatureSpec, cfg: Config, *, mc_dropout: bool = False) -> None:
        self.spec = spec
        self.cfg = cfg
        self.mc_dropout = mc_dropout
        self.model_: Any | None = None
        self.history_: dict[str, list[float]] = {}

        self.sequence_columns = np.asarray(spec.sequence_columns, dtype=int)
        self.static_columns = np.asarray(spec.static_columns, dtype=int)
        self.state_columns = np.asarray(spec.state_columns, dtype=int)

        if len(self.sequence_columns) == 0:
            raise LSTMError(
                "no time-varying columns to feed the recurrent layers; every feature "
                "was classified as static or state identity"
            )

    # -- Forecaster protocol -------------------------------------------------- #

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> PooledLSTM:
        """Train on one fold, stopping early on the supplied validation block.

        The validation block comes from :mod:`src.splits` and is a genuine future
        time window, not a random subset. Early stopping on a randomly-carved set
        would be measuring memorisation, not generalisation.
        """
        import keras

        keras.utils.set_random_seed(self.cfg.project.seed)

        self.model_ = self._build(timesteps=X.shape[1], n_features=X.shape[2])
        lstm = self.cfg.model.lstm

        callbacks = []
        if validation is not None:
            callbacks.append(
                keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=lstm.early_stopping_patience,
                    restore_best_weights=True,
                )
            )

        history = self.model_.fit(
            X,
            y,
            validation_data=validation,
            epochs=lstm.max_epochs,
            batch_size=lstm.batch_size,
            callbacks=callbacks,
            shuffle=False,
            verbose=0,
        )
        self.history_ = {key: [float(v) for v in values] for key, values in history.history.items()}
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Point forecasts on the transformed target scale."""
        if self.model_ is None:
            raise LSTMError("predict called before fit")
        if self.mc_dropout:
            return np.asarray(self.model_(X, training=True), dtype=float).ravel()
        return np.asarray(self.model_.predict(X, verbose=0), dtype=float).ravel()

    # -- diagnostics ---------------------------------------------------------- #

    def predict_samples(self, X: np.ndarray, n_samples: int) -> np.ndarray:
        """Repeat prediction with dropout active, giving ``(n_samples, n_rows)``.

        Useful as a sanity check on interval width, and as the fallback if
        conformal calibration turns out to have too few residuals to be stable.
        """
        if self.model_ is None:
            raise LSTMError("predict_samples called before fit")
        return np.stack(
            [np.asarray(self.model_(X, training=True), dtype=float).ravel()
             for _ in range(n_samples)]
        )

    @property
    def diagnostics(self) -> dict[str, float]:
        """Final losses and the train/validation gap.

        A widening gap is the signal to shrink the network, so it is recorded per
        fold rather than left in a console log that nobody keeps.
        """
        if not self.history_:
            return {}
        train = self.history_.get("loss", [])
        val = self.history_.get("val_loss", [])
        out: dict[str, float] = {"epochs_run": float(len(train))}
        if train:
            out["final_train_loss"] = train[-1]
            out["best_train_loss"] = min(train)
        if val:
            out["final_val_loss"] = val[-1]
            out["best_val_loss"] = min(val)
            out["best_epoch"] = float(int(np.argmin(val)) + 1)
        if train and val:
            out["train_val_gap"] = val[-1] - train[-1]
        return out

    # -- architecture --------------------------------------------------------- #

    def _build(self, timesteps: int, n_features: int) -> Any:
        """Assemble the network.

        Shape of it::

            sequence columns  -> LSTM stack ---.
            state one-hot @t  -> Dense(embed) --+-> Dense -> Dense(1)
            static @t         ------------------'
        """
        import keras
        from keras import layers

        column_select, last_timestep = register_serializable_layers()
        lstm = self.cfg.model.lstm
        inputs = keras.Input(shape=(timesteps, n_features), name="features")

        sequence = column_select(
            list(self.sequence_columns), axis=2, name="sequence_columns"
        )(inputs)

        recurrent = sequence
        for depth in range(lstm.layers):
            recurrent = layers.LSTM(
                lstm.units,
                return_sequences=depth < lstm.layers - 1,
                dropout=lstm.dropout,
                recurrent_dropout=lstm.recurrent_dropout,
                name=f"lstm_{depth}",
            )(recurrent)

        # Static and identity columns are constant across the window, so only the
        # forecast origin's row is read.
        last = last_timestep(name="at_origin")(inputs)

        merged = [recurrent]
        if len(self.state_columns):
            one_hot = column_select(
                list(self.state_columns), axis=1, name="state_one_hot"
            )(last)
            # A bias-free dense layer on a one-hot vector is exactly an embedding
            # lookup, and keeps the whole model a single tensor input.
            merged.append(
                layers.Dense(
                    lstm.state_embedding_dim, use_bias=False, name="state_embedding"
                )(one_hot)
            )
        if len(self.static_columns):
            merged.append(
                column_select(list(self.static_columns), axis=1, name="static_columns")(last)
            )

        joined = layers.Concatenate(name="join")(merged) if len(merged) > 1 else recurrent
        hidden = layers.Dense(max(lstm.units // 4, 8), activation="relu", name="head")(joined)
        hidden = layers.Dropout(lstm.dropout, name="head_dropout")(hidden)
        output = layers.Dense(1, name="prediction")(hidden)

        model = keras.Model(inputs=inputs, outputs=output, name="pooled_lstm")
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lstm.learning_rate),
            # The target is already log(cases per 100k + 1), so squared error here
            # is squared error on the log scale. Raw-count MSE would let a single
            # outbreak month dominate every gradient.
            loss="mse",
            metrics=["mae"],
        )
        return model


def pooled_lstm(
    spec: FeatureSpec, cfg: Config, *, mc_dropout: bool = False
) -> ForecasterFactory:
    """Factory for :func:`~src.evaluate.run_experiment`.

    Wrapped in :class:`~src.models.scaling.StandardScaled`, because a network needs
    standardised inputs and the scaler must be fitted per fold.
    """
    from src.models.scaling import StandardScaled

    return lambda: StandardScaled(PooledLSTM(spec, cfg, mc_dropout=mc_dropout))


_: type[Forecaster] = PooledLSTM
