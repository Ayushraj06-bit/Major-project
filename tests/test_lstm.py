"""The pooled LSTM: architecture constraints, determinism, and harness fit.

Kept small and on tiny fixtures. These check the design constraints hold, not that
the model is accurate, which is a data question rather than a code question.
"""

from __future__ import annotations

import dataclasses
import inspect

import numpy as np
import pandas as pd
import pytest

from src.config import Config
from src.evaluate import run_experiment
from src.features import STATE_PREFIX, build_features
from src.models import Forecaster
from src.models.lstm import PooledLSTM, pooled_lstm
from src.models.scaling import StandardScaled
from src.panel import complete_index

pytest.importorskip("keras", reason="LSTM tests need keras")


@pytest.fixture
def tiny_cfg(cfg: Config) -> Config:
    """Small everything, so a fold trains in a second or two."""
    return dataclasses.replace(
        cfg,
        data=dataclasses.replace(
            cfg.data,
            start_date=pd.Timestamp("2012-01-01").date(),
            end_date=pd.Timestamp("2017-12-31").date(),
        ),
        features=dataclasses.replace(
            cfg.features, sequence_length=3, lags=(1, 2, 12), rolling_windows=(3,)
        ),
        model=dataclasses.replace(
            cfg.model,
            lstm=dataclasses.replace(
                cfg.model.lstm, units=8, layers=1, max_epochs=3,
                early_stopping_patience=2, state_embedding_dim=2,
            ),
        ),
        split=dataclasses.replace(
            cfg.split, n_folds=2, initial_train_size=40, test_size=8, step=8
        ),
    )


@pytest.fixture
def tiny_data(tiny_cfg: Config):
    index = complete_index(tiny_cfg)
    month = index.get_level_values("date").month.to_numpy()
    rng = np.random.default_rng(3)
    seasonal = np.sin(2 * np.pi * month / 12)
    panel = pd.DataFrame(
        {
            "cases": np.abs(50 + 30 * seasonal + rng.normal(0, 3, len(index))),
            "rainfall": 100 + 80 * seasonal,
            "temperature": 27 + 4 * seasonal,
            "humidity": 70 + 10 * seasonal,
            "search_interest": 40 + 20 * seasonal,
            "population": 3.0e7,
            "population_density": 850.0,
        },
        index=index,
    )
    return build_features(panel, tiny_cfg)


# --------------------------------------------------------------------------- #
# Architecture constraints
# --------------------------------------------------------------------------- #


def test_state_identity_is_a_learned_embedding_after_the_recurrence(
    tiny_data, tiny_cfg: Config
) -> None:
    """Constraint: state joins after the recurrent layers, not inside them."""
    X, y, spec = tiny_data
    model = PooledLSTM(spec, tiny_cfg)
    model.fit(X[:60], y[:60])

    names = [layer.name for layer in model.model_.layers]
    assert "state_embedding" in names

    embedding = model.model_.get_layer("state_embedding")
    assert embedding.use_bias is False, "a bias would stop it being an embedding lookup"
    assert embedding.units == tiny_cfg.model.lstm.state_embedding_dim

    # The one-hot columns must not reach the recurrent input.
    assert set(spec.state_columns).isdisjoint(spec.sequence_columns)


def test_static_features_join_at_the_dense_layer_not_as_sequence_input(
    tiny_data, tiny_cfg: Config
) -> None:
    """Repeating population density across timesteps teaches an LSTM nothing."""
    _, _, spec = tiny_data
    density = spec.columns.index("population_density")
    assert density in spec.static_columns
    assert density not in spec.sequence_columns


def test_state_columns_are_one_hot_and_cover_every_state(tiny_data) -> None:
    X, _, spec = tiny_data
    names = [spec.columns[i] for i in spec.state_columns]
    assert all(name.startswith(STATE_PREFIX) for name in names)
    surviving = set(spec.sample_index.get_level_values("state"))
    assert len(names) == len(surviving) + len(spec.dropped_states)

    one_hot = X[:, -1, list(spec.state_columns)]
    np.testing.assert_allclose(one_hot.sum(axis=1), 1.0)


def test_network_stays_small(tiny_data, tiny_cfg: Config) -> None:
    """The reference paper's 1000 units would memorise a dataset this size."""
    X, y, spec = tiny_data
    model = PooledLSTM(spec, tiny_cfg).fit(X[:60], y[:60])
    assert model.model_.count_params() < 20_000


def test_more_than_two_layers_is_rejected_by_config(cfg: Config) -> None:
    from src.config import ConfigError

    with pytest.raises(ConfigError, match="too deep"):
        dataclasses.replace(cfg.model.lstm, layers=4)


# --------------------------------------------------------------------------- #
# Determinism and dropout
# --------------------------------------------------------------------------- #


def test_training_is_deterministic_under_a_fixed_seed(tiny_data, tiny_cfg: Config) -> None:
    X, y, spec = tiny_data
    first = PooledLSTM(spec, tiny_cfg).fit(X[:80], y[:80]).predict(X[80:100])
    second = PooledLSTM(spec, tiny_cfg).fit(X[:80], y[:80]).predict(X[80:100])
    np.testing.assert_allclose(first, second, rtol=1e-5, atol=1e-6)


def test_dropout_is_off_at_inference_by_default(tiny_data, tiny_cfg: Config) -> None:
    X, y, spec = tiny_data
    model = PooledLSTM(spec, tiny_cfg).fit(X[:80], y[:80])
    np.testing.assert_allclose(model.predict(X[80:100]), model.predict(X[80:100]))


def test_mc_dropout_flag_makes_prediction_stochastic(tiny_data, tiny_cfg: Config) -> None:
    """Exposed for later use as a cross-check on conformal interval width."""
    X, y, spec = tiny_data
    model = PooledLSTM(spec, tiny_cfg, mc_dropout=True).fit(X[:80], y[:80])
    assert not np.allclose(model.predict(X[80:100]), model.predict(X[80:100]))

    samples = model.predict_samples(X[80:100], n_samples=5)
    assert samples.shape == (5, 20)
    assert samples.std(axis=0).max() > 0


# --------------------------------------------------------------------------- #
# Harness fit
# --------------------------------------------------------------------------- #


def test_lstm_satisfies_the_forecaster_protocol(tiny_data, tiny_cfg: Config) -> None:
    _, _, spec = tiny_data
    assert isinstance(pooled_lstm(spec, tiny_cfg)(), Forecaster)


def test_factory_wraps_in_the_fold_scaler(tiny_data, tiny_cfg: Config) -> None:
    _, _, spec = tiny_data
    assert isinstance(pooled_lstm(spec, tiny_cfg)(), StandardScaled)


def test_runs_through_the_unmodified_harness(tiny_data, tiny_cfg: Config) -> None:
    """The whole point of Phase 4: no special-casing for the LSTM."""
    _, _, spec = tiny_data
    result = run_experiment(pooled_lstm(spec, tiny_cfg), tiny_data, tiny_cfg, "lstm_probe")

    assert len(result.folds) == tiny_cfg.split.n_folds
    assert np.isfinite(result.primary)
    assert result.std["mae_log"] >= 0.0


def test_training_diagnostics_are_recorded_per_fold(tiny_data, tiny_cfg: Config) -> None:
    """The review gate asks whether the train/val gap is widening; record it."""
    _, _, spec = tiny_data
    result = run_experiment(pooled_lstm(spec, tiny_cfg), tiny_data, tiny_cfg, "lstm_probe")
    for fold in result.folds:
        assert {"epochs_run", "final_train_loss", "final_val_loss", "train_val_gap"} <= set(
            fold.diagnostics
        )


def test_early_stopping_uses_the_validation_block_from_splits(
    tiny_data, tiny_cfg: Config
) -> None:
    """Not a tail carve of X: X is state-major, so its tail is one state."""
    X, y, spec = tiny_data
    model = PooledLSTM(spec, tiny_cfg)
    model.fit(X[:80], y[:80], (X[80:100], y[80:100]))
    assert "val_loss" in model.history_


def test_no_second_training_loop_exists() -> None:
    """The review gate: the fold loop lives in evaluate and nowhere else."""
    import src.models.lstm as module

    body = inspect.getsource(module).split('"""', 2)[-1]
    for forbidden in ("rolling_origin", "for fold", "compute_metrics"):
        assert forbidden not in body, f"{forbidden} appeared in src/models/lstm.py"
