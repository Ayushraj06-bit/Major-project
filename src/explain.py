"""SHAP attribution over the frozen production model.

Two consumers, one implementation. The same attribution matrix answers "why did
the model say that?" for a single prediction, and "which inputs matter?" for
feature selection. Computing them twice would let the explanation on the dashboard
drift from the ranking that shaped the model.

**Why KernelExplainer rather than DeepExplainer.** DeepExplainer does not support
TensorFlow 2.x recurrent layers, and has not for years. Rather than pin a
TensorFlow version around a SHAP limitation, the model is wrapped so SHAP only
ever sees a flat 2-D matrix and never learns it is explaining an LSTM::

    def _predict_flat(X_flat):
        return model.predict(X_flat.reshape(-1, T, F)).ravel()

That is slow but version-proof. ``GradientExplainer`` sits behind
``explain.explainer: gradient`` as a faster path for builds where it works.

**Attributions read in domain terms.** A raw SHAP matrix is
``rows x (timesteps * features)`` and indexed by position, which nobody can read.
:class:`~src.features.FeatureSpec` carries the provenance of every column, so
output says "rainfall lag-3" and "neighbouring cases lag-1", never "feature_47".

**This module never trains anything.** It loads the frozen artifact from
:mod:`src.production` and explains that. If it built its own model, the dashboard
would show an explanation of a different model from the one making the forecast.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.artifacts import load_run, save_run
from src.config import Config
from src.features import (
    CALENDAR_VARIABLE,
    STATE_VARIABLE,
    TARGET_LEVEL_COLUMN,
    TRANSFORM_CYCLIC,
    TRANSFORM_IDENTITY,
    TRANSFORM_LAG,
    TRANSFORM_ROLLING,
    TRANSFORM_SPATIAL_LAG,
    TRANSFORM_STATIC,
    FeatureOrigin,
    FeatureSpec,
)

#: Run name the cached attributions are stored under.
SHAP_RUN = "shap"

#: Readable labels for the pseudo-variables that come from no data source.
PSEUDO_VARIABLE_LABELS = {
    CALENDAR_VARIABLE: "calendar seasonality",
    STATE_VARIABLE: "state identity",
    TARGET_LEVEL_COLUMN: "case rate",
}


class ExplainError(RuntimeError):
    """Raised when attributions cannot be computed or read back."""


@contextmanager
def _seeded(seed: int) -> Iterator[None]:
    """Run with a fixed global NumPy seed, then restore what was there.

    SHAP samples internally from NumPy's global generator, so attributions would
    otherwise depend on whatever last touched it. That matters beyond tests: these
    values are cached and shown on a dashboard, and an explanation that changes
    between runs of the same model is not an explanation.

    The previous state is restored so that seeding here cannot silently make some
    later caller reproducible when it is not.
    """
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


@dataclass(frozen=True)
class Attribution:
    """SHAP values for a set of predictions, with everything needed to read them.

    Attributes:
        values: ``(n_rows, n_features)`` attributions, already summed over the time
            axis. A ``timesteps x features`` grid is unreadable and nobody has ever
            usefully interpreted one.
        base_value: The explainer's expected output, so contributions sum to the
            prediction.
        columns: Feature names matching the second axis of ``values``.
        sample_index: ``(state, date)`` of each explained row.
        explainer: Which backend produced these.
    """

    values: np.ndarray
    base_value: float
    columns: tuple[str, ...]
    sample_index: pd.MultiIndex
    explainer: str

    def frame(self) -> pd.DataFrame:
        """Attributions as a labelled frame, one row per explained prediction."""
        return pd.DataFrame(self.values, columns=list(self.columns), index=self.sample_index)

    def global_importance(self, spec: FeatureSpec) -> pd.DataFrame:
        """Mean absolute attribution per column, most important first.

        Mean *absolute* value, because a driver that pushes predictions up in the
        monsoon and down in winter matters a great deal while averaging to nothing.
        """
        importance = np.abs(self.values).mean(axis=0)
        return (
            pd.DataFrame(
                {
                    "column": self.columns,
                    "readable": [describe_column(spec.origins[c]) for c in self.columns],
                    "raw_variable": [spec.origins[c].raw_variable for c in self.columns],
                    "mean_abs_shap": importance,
                }
            )
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )

    def by_raw_variable(self, spec: FeatureSpec) -> pd.DataFrame:
        """Attribution summed back to the raw driver, most important first.

        This is the view the report and the dashboard show. "Rainfall matters"
        is a sentence a public-health reader can act on; "rainfall_roll3_mean at
        t-7 contributes 0.0041" is not.
        """
        frame = self.global_importance(spec)
        return (
            frame.groupby("raw_variable", as_index=False)["mean_abs_shap"]
            .sum()
            .assign(label=lambda f: f["raw_variable"].map(_variable_label))
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )

    def top_drivers(self, spec: FeatureSpec, row: int, k: int = 3) -> list[tuple[str, float]]:
        """The ``k`` largest contributors to one prediction, in domain terms.

        This is what the recommendation layer quotes: "Top drivers: rainfall lag-3,
        humidity lag-2".
        """
        contributions = self.values[row]
        order = np.argsort(-np.abs(contributions))[:k]
        return [
            (describe_column(spec.origins[self.columns[position]]), float(contributions[position]))
            for position in order
        ]


# --------------------------------------------------------------------------- #
# Readable names
# --------------------------------------------------------------------------- #


def describe_column(origin: FeatureOrigin) -> str:
    """Turn a derived column's provenance into a phrase a reader understands.

    Review gate: attributions must read in domain terms. This is where that
    happens, and it works off the recorded transform rather than by parsing the
    column name, so a renamed column cannot silently produce nonsense.
    """
    variable = _variable_label(origin.raw_variable)

    # The target's own autoregressive terms carry raw_variable "cases", the same as
    # the raw count lags, so both would render as "cases lag-6". They are different
    # quantities -- one is a count, one is the log case rate -- and a reader acting
    # on an attribution needs to know which.
    if origin.column.startswith(TARGET_LEVEL_COLUMN):
        variable = PSEUDO_VARIABLE_LABELS[TARGET_LEVEL_COLUMN]
        return f"{variable} (current)" if origin.lag == 0 else f"{variable} lag-{origin.lag}"

    if origin.transform == TRANSFORM_CYCLIC:
        return f"{variable} ({'sine' if origin.column.endswith('sin') else 'cosine'})"
    if origin.transform == TRANSFORM_STATIC:
        if origin.raw_variable == STATE_VARIABLE:
            return f"state identity: {origin.column.removeprefix('state_is_')}"
        return variable
    if origin.transform == TRANSFORM_SPATIAL_LAG:
        return f"neighbouring {variable} lag-{origin.lag}"
    if origin.transform == TRANSFORM_ROLLING:
        return f"{variable} {origin.window}-period {origin.agg}"
    if origin.transform == TRANSFORM_LAG:
        return f"{variable} (current)" if origin.lag == 0 else f"{variable} lag-{origin.lag}"
    if origin.transform == TRANSFORM_IDENTITY:
        return f"{variable} (current)"
    return origin.column


def _variable_label(raw_variable: str) -> str:
    """Human wording for a raw panel variable or pseudo-variable."""
    if raw_variable in PSEUDO_VARIABLE_LABELS:
        return PSEUDO_VARIABLE_LABELS[raw_variable]
    return raw_variable.replace("_", " ")


# --------------------------------------------------------------------------- #
# Computing attributions
# --------------------------------------------------------------------------- #


def explain(
    predictor: Any,
    X: np.ndarray,
    spec: FeatureSpec,
    cfg: Config,
    *,
    rows: np.ndarray | None = None,
) -> Attribution:
    """Attribute predictions to features, aggregated over the time axis.

    Args:
        predictor: The fitted model from the production artifact.
        X: ``(n, timesteps, n_features)`` scaled input the model expects.
        spec: Feature spec describing ``X``.
        cfg: Loaded configuration; ``explain`` supplies the budget.
        rows: Which rows to explain. A capped, evenly-spaced sample by default,
            because KernelExplainer cost is linear in rows and explaining a whole
            panel buys no extra insight.

    Returns:
        The :class:`Attribution`.

    Raises:
        ExplainError: SHAP is unavailable, or the configured explainer fails.
    """
    try:
        import shap
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ExplainError("SHAP is not installed: pip install shap") from exc

    rows = _rows_to_explain(len(X), cfg) if rows is None else np.asarray(rows, dtype=int)
    timesteps, n_features = X.shape[1], X.shape[2]

    forward = _fast_forward(predictor)

    def predict_flat(flat: np.ndarray) -> np.ndarray:
        """The wrapper that hides the recurrent shape from SHAP."""
        reshaped = np.asarray(flat, dtype=np.float32).reshape(-1, timesteps, n_features)
        return forward(reshaped)

    flat = X.reshape(len(X), -1)
    # Seeded explicitly: shap.sample also draws from the global generator.
    background = shap.sample(
        flat,
        min(cfg.explain.background_samples, len(flat)),
        random_state=cfg.project.seed,
    )

    with _seeded(cfg.project.seed):
        if cfg.explain.explainer == "gradient":
            values, base = _gradient_values(shap, predictor, X, rows, cfg)
        else:
            explainer = shap.KernelExplainer(predict_flat, background)
            raw = explainer.shap_values(
                flat[rows], nsamples=cfg.explain.nsamples, silent=True
            )
            values = np.asarray(raw, dtype=float).reshape(len(rows), timesteps, n_features)
            base = float(np.ravel(explainer.expected_value)[0])

    return Attribution(
        # Summed over the time axis: one number per feature, per prediction.
        values=values.sum(axis=1),
        base_value=base,
        columns=spec.columns,
        sample_index=spec.sample_index[rows],
        explainer=cfg.explain.explainer,
    )


def _fast_forward(predictor: Any) -> Any:
    """Pick the cheapest correct way to run this model on many small batches.

    KernelExplainer calls the prediction function thousands of times with batches
    of a few hundred rows. Keras ``.predict()`` is built for one large pass: it
    constructs a dataset and a traced function per call, and that fixed cost
    dominates when the batches are small. Calling the model directly skips it and
    returns identical values, measured at roughly 1.5x end to end here.

    Anything that is not a Keras model — a scikit-learn estimator, a stub in a
    test — falls back to ``.predict()``.
    """
    try:
        import keras

        if isinstance(predictor, keras.Model):
            return lambda batch: np.asarray(
                predictor(batch, training=False), dtype=float
            ).ravel()
    except ImportError:  # pragma: no cover - environment dependent
        pass

    return lambda batch: np.asarray(predictor.predict(batch, verbose=0), dtype=float).ravel()


def _gradient_values(
    shap: Any, predictor: Any, X: np.ndarray, rows: np.ndarray, cfg: Config
) -> tuple[np.ndarray, float]:
    """GradientExplainer path, far faster where the build supports it.

    Raises:
        ExplainError: the build cannot differentiate through these layers, which is
            common and is why kernel is the default.
    """
    try:
        background = X[
            np.linspace(0, len(X) - 1, min(cfg.explain.background_samples, len(X)), dtype=int)
        ]
        explainer = shap.GradientExplainer(predictor, background)
        raw = explainer.shap_values(X[rows])
    except Exception as exc:  # noqa: BLE001 - reported, with the fallback named
        raise ExplainError(
            f"GradientExplainer failed on this build ({exc}). Set "
            "explain.explainer: kernel, which is slower but version-proof."
        ) from exc
    return np.asarray(raw, dtype=float).reshape(len(rows), X.shape[1], X.shape[2]), 0.0


def _rows_to_explain(n_rows: int, cfg: Config) -> np.ndarray:
    """Evenly-spaced sample of rows, bounded by ``explain.max_explained_rows``.

    Evenly spaced rather than random so the sample spans the whole panel: every
    state and the full date range, rather than a lucky draw of one season.
    """
    limit = min(cfg.explain.max_explained_rows, n_rows)
    return np.linspace(0, n_rows - 1, limit, dtype=int)


# --------------------------------------------------------------------------- #
# Feature selection
# --------------------------------------------------------------------------- #


def select_features(
    attribution: Attribution, spec: FeatureSpec, cfg: Config
) -> tuple[str, ...]:
    """The global top-k columns by mean absolute attribution.

    This is what feeds back into config as an ablation variant. It is deliberately
    **global**, not per-state, even though the reference paper selects per state:
    a pooled model has one input matrix, so per-state column sets cannot coexist in
    it. Per-state rankings are still computed by :func:`per_state_ranking` and
    reported as a finding — which drivers matter where is interesting in its own
    right, and it is the honest way to keep that analysis without pretending the
    model can act on it.

    State-identity columns are excluded from the ranking. They are not drivers
    competing for a slot; :func:`~src.features.apply_selection` keeps them
    regardless.
    """
    importance = attribution.global_importance(spec)
    drivers = importance[importance["raw_variable"] != STATE_VARIABLE]
    return tuple(drivers.head(cfg.explain.top_k_features)["column"])


def per_state_ranking(attribution: Attribution, spec: FeatureSpec) -> pd.DataFrame:
    """Mean absolute attribution per column, per state.

    Reported as a finding rather than used for selection — see
    :func:`select_features`. "Rainfall dominates in Kerala, case history dominates
    in Delhi" is a result worth writing down.

    State-identity columns are excluded. They rank highly and mean nothing here: a
    large attribution on "state is Odisha" for a Kerala row is the model registering
    that the flag is *off*, which is real arithmetic but not a driver anyone can act
    on, and it crowds the genuine drivers out of the top of the table.
    """
    drivers = [
        column
        for column in attribution.columns
        if spec.origins[column].raw_variable != STATE_VARIABLE
    ]
    ranked = attribution.frame().abs()[drivers].groupby(level="state").mean().T
    ranked.index.name = "column"
    return (
        ranked.reset_index()
        .assign(readable=lambda f: f["column"].map(lambda c: describe_column(spec.origins[c])))
        .set_index(["column", "readable"])
    )


# --------------------------------------------------------------------------- #
# Caching
# --------------------------------------------------------------------------- #


def save_attribution(
    attribution: Attribution, spec: FeatureSpec, *, name: str = SHAP_RUN
) -> None:
    """Cache attributions so the dashboard never recomputes them.

    KernelExplainer is minutes-to-hours; a dashboard that recomputes on every page
    load is not a dashboard. Stored through the same artifact store as everything
    else, so it is read the same way.
    """
    save_run(
        name,
        overwrite=True,
        shap_values=attribution.values,
        shap_meta={
            "base_value": attribution.base_value,
            "columns": list(attribution.columns),
            "explainer": attribution.explainer,
            "readable": [describe_column(spec.origins[c]) for c in attribution.columns],
            "sample_index": [
                [str(state), pd.Timestamp(date).isoformat()]
                for state, date in attribution.sample_index
            ],
        },
        importance=attribution.global_importance(spec),
        by_variable=attribution.by_raw_variable(spec),
    )


def load_attribution(name: str = SHAP_RUN) -> Attribution:
    """Read cached attributions back.

    Raises:
        ExplainError: nothing has been cached yet.
    """
    try:
        payload = load_run(name)
    except FileNotFoundError as exc:
        raise ExplainError(
            f"no cached attributions under run {name!r}; run scripts/run_shap.py first"
        ) from exc

    meta = payload["shap_meta"]
    pairs = [(state, pd.Timestamp(date)) for state, date in meta["sample_index"]]
    return Attribution(
        values=np.asarray(payload["shap_values"], dtype=float),
        base_value=float(meta["base_value"]),
        columns=tuple(meta["columns"]),
        sample_index=pd.MultiIndex.from_tuples(pairs, names=["state", "date"]),
        explainer=str(meta["explainer"]),
    )
