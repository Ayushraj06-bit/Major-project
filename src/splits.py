"""Rolling-origin cross-validation over a pooled panel.

Every fold is a strictly time-ordered cut: train on everything up to a date, test
on the window after it. There is no shuffle option, because random splitting
leaks future observations into training and invalidates every metric downstream.

**Why this takes a sample index rather than a sample count.** In a pooled panel
the samples are ordered by state, then by date — all of Kerala, then all of
Odisha, and so on. Splitting positionally on ``n_samples`` would therefore hold
out *states*, not *time*: train and test would span the same date range, and the
model would be scored on generalising across states while having already seen the
test period. That is a different experiment wearing this one's name, and it
inflates the result. Folds are cut on dates, and every state contributes rows to
both sides of every cut.

Each fold yields three usable blocks, in time order::

    [ fit | val | embargo | test ]

``val`` is the tail of the training window, held out for conformal calibration and
for any model wanting an early-stopping set. ``embargo`` is dropped entirely: a
training sample at origin ``T`` carries a label from ``T + horizon``, so without a
gap the last training labels would come from inside the test window.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import Config


class SplitError(RuntimeError):
    """Raised when the configured fold geometry does not fit the data."""


@dataclass(frozen=True)
class Fold:
    """One rolling-origin fold, as positional indices into the sample arrays.

    Attributes:
        number: Zero-based fold index.
        fit: Rows to fit the model on.
        val: Rows held out from fitting, for interval calibration or early stopping.
        test: Rows to score on.
        fit_end: Last forecast origin in ``fit``.
        val_end: Last forecast origin in ``val``.
        test_start: First forecast origin in ``test``.
        test_end: Last forecast origin in ``test``.
        embargo: Periods dropped between ``val`` and ``test``.
    """

    number: int
    fit: np.ndarray
    val: np.ndarray
    test: np.ndarray
    fit_end: pd.Timestamp
    val_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    embargo: int

    @property
    def train(self) -> np.ndarray:
        """``fit`` and ``val`` together, for models wanting the whole history."""
        return np.concatenate([self.fit, self.val])

    def describe(self) -> str:
        """One-line summary, for logs and the run record."""
        return (
            f"fold {self.number}: fit {len(self.fit)} (to {self.fit_end.date()}) | "
            f"val {len(self.val)} (to {self.val_end.date()}) | "
            f"embargo {self.embargo} | "
            f"test {len(self.test)} ({self.test_start.date()} to {self.test_end.date()})"
        )


def rolling_origin(
    sample_index: pd.MultiIndex | pd.DatetimeIndex,
    cfg: Config,
    *,
    horizon: int | None = None,
) -> Iterator[Fold]:
    """Yield expanding-window folds, cut on dates.

    Args:
        sample_index: Forecast origins, one per sample. Either the ``(state, date)``
            index from a :class:`~src.features.FeatureSpec`, or a plain
            ``DatetimeIndex`` for a single series.
        cfg: Loaded configuration; ``split`` supplies the geometry.
        horizon: Forecast lead time, which sets the embargo width. Defaults to the
            first entry of ``forecast.horizons``.

    Yields:
        One :class:`Fold` per configured fold, in chronological order.

    Raises:
        SplitError: the index is unusable, or the data cannot support the
            configured number of folds.
    """
    horizon = horizon if horizon is not None else cfg.forecast.horizons[0]
    dates = _origin_dates(sample_index)
    periods = pd.DatetimeIndex(sorted(pd.unique(dates)))

    _require_geometry_fits(len(periods), cfg, horizon)

    split = cfg.split
    validation_fraction = cfg.model.lstm.validation_fraction

    for number in range(split.n_folds):
        cut = split.initial_train_size + number * split.step
        train_periods = periods[:cut]
        test_periods = periods[cut : cut + split.test_size]

        usable = train_periods[: len(train_periods) - horizon] if horizon else train_periods
        n_val = max(1, int(round(len(usable) * validation_fraction)))
        fit_periods = usable[: len(usable) - n_val]
        val_periods = usable[len(usable) - n_val :]

        if len(fit_periods) == 0:
            raise SplitError(
                f"fold {number}: validation_fraction {validation_fraction} leaves no "
                f"periods to fit on out of {len(usable)} usable training periods"
            )

        yield Fold(
            number=number,
            fit=_positions(dates, fit_periods),
            val=_positions(dates, val_periods),
            test=_positions(dates, test_periods),
            fit_end=fit_periods[-1],
            val_end=val_periods[-1],
            test_start=test_periods[0],
            test_end=test_periods[-1],
            embargo=horizon,
        )


def n_available_folds(n_periods: int, cfg: Config) -> int:
    """How many folds a given number of periods can support.

    Reported in the geometry error so the fix needs no arithmetic.
    """
    split = cfg.split
    spare = n_periods - split.initial_train_size - split.test_size
    return 0 if spare < 0 else spare // split.step + 1


def _origin_dates(sample_index: pd.MultiIndex | pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Extract the forecast-origin date of every sample."""
    if isinstance(sample_index, pd.MultiIndex):
        if "date" not in list(sample_index.names or []):
            raise SplitError(
                f"sample index must carry a 'date' level, got names {list(sample_index.names)}"
            )
        return pd.DatetimeIndex(sample_index.get_level_values("date"))
    if isinstance(sample_index, pd.DatetimeIndex):
        return sample_index
    raise SplitError(
        "rolling_origin needs the forecast origin of every sample — pass a "
        "FeatureSpec.sample_index or a DatetimeIndex, not a sample count. Splitting a "
        "pooled panel positionally holds out states rather than time."
    )


def _positions(dates: pd.DatetimeIndex, wanted: Sequence[pd.Timestamp]) -> np.ndarray:
    """Positional indices of every sample whose origin falls in ``wanted``."""
    return np.flatnonzero(dates.isin(pd.DatetimeIndex(wanted)))


def _require_geometry_fits(n_periods: int, cfg: Config, horizon: int) -> None:
    """Fail loudly rather than silently yielding fewer folds than configured.

    Quietly running two folds when the config asks for four would still be
    reported as "mean +/- std across 4 folds", which is simply false.
    """
    split = cfg.split
    needed = split.initial_train_size + (split.n_folds - 1) * split.step + split.test_size
    if n_periods < needed:
        raise SplitError(
            f"{n_periods} distinct periods cannot support {split.n_folds} folds of "
            f"initial_train_size={split.initial_train_size}, step={split.step}, "
            f"test_size={split.test_size} (needs {needed}); at most "
            f"{n_available_folds(n_periods, cfg)} fold(s) fit"
        )
    if split.initial_train_size <= horizon:
        raise SplitError(
            f"split.initial_train_size ({split.initial_train_size}) must exceed the "
            f"horizon ({horizon}), or the embargo consumes the whole training window"
        )
