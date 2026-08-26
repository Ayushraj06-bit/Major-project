"""The project's single caching layer.

Every expensive call — a network fetch, a gridded-climate aggregation, a slow
join — is decorated with :func:`cached`. The decorator owns all cache existence
checking, so no other module anywhere in the project inspects the filesystem to
decide whether it needs to recompute something.

Cached results are Parquet files under ``paths.data_interim``. The cache key
combines the caller-supplied name with a fingerprint of the bound arguments, so
``fetch_cases(state="Kerala")`` and ``fetch_cases(state="Odisha")`` cannot
collide — a bare name would silently return one state's data for the other.
"""

from __future__ import annotations

import hashlib
import inspect
import os
import re
from collections.abc import Callable, Mapping, Sequence
from datetime import date
from functools import wraps
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from src.config import load_config

CACHE_DISABLED_ENV_VAR = "DENGUE_CACHE_DISABLED"
CACHE_SUFFIX = ".parquet"

_KEY_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
_FINGERPRINT_LENGTH = 12
_SEPARATOR = "__"


class CachedFunction(Protocol):
    """A function wrapped by :func:`cached`, plus the cache controls it gains."""

    def __call__(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """Return the cached result, computing and storing it on a miss."""
        ...

    def cache_path(self, *args: Any, **kwargs: Any) -> Path:
        """Where the result for these arguments is or would be stored."""

    def clear_cache(self) -> int:
        """Delete every cached result for this function. Returns the file count."""


def cached(
    key: str, *, key_args: Sequence[str] = ()
) -> Callable[[Callable[..., pd.DataFrame]], CachedFunction]:
    """Memoise a DataFrame-returning function to Parquet under ``data/interim/``.

    On a hit the stored Parquet is read and the wrapped function never runs. On a
    miss the function runs and its result is written before being returned. Set
    ``$DENGUE_CACHE_DISABLED=1`` to force recomputation without editing code.

    Arguments must be primitives or simple containers of primitives, because they
    are fingerprinted by value. Passing a DataFrame or an array raises — those
    have no stable representation, so the cache could not tell two different
    inputs apart and would return the wrong data.

    Args:
        key: Stable snake_case name for this function's cache family. It prefixes
            every file, so ``data/interim/`` stays readable by eye.
        key_args: Parameter names whose values are spelled out in the filename
            rather than only hashed. Use when one function serves several
            families — a generic source fetcher keyed by ``source_name`` yields
            ``source__cases__1a2b3c.parquet`` instead of an opaque digest.

    Returns:
        The wrapped function, with ``cache_path`` and ``clear_cache`` attached.

    Raises:
        ValueError: ``key`` is not snake_case, or ``key_args`` names a parameter
            the function does not accept.
        TypeError: the wrapped function returned something other than a DataFrame.
    """
    if not _KEY_PATTERN.match(key):
        raise ValueError(
            f"cache key must be snake_case matching {_KEY_PATTERN.pattern!r}, got {key!r}"
        )

    def decorator(func: Callable[..., pd.DataFrame]) -> CachedFunction:
        signature = inspect.signature(func)
        unknown = [name for name in key_args if name not in signature.parameters]
        if unknown:
            raise ValueError(
                f"@cached({key!r}): key_args {unknown} are not parameters of {func.__name__}"
            )

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> pd.DataFrame:
            path = _cache_path(key, signature, args, kwargs, key_args)
            if not _cache_disabled() and path.is_file():
                return pd.read_parquet(path)

            result = func(*args, **kwargs)
            if not isinstance(result, pd.DataFrame):
                raise TypeError(
                    f"@cached({key!r}) expects {func.__name__} to return a DataFrame, "
                    f"got {type(result).__name__}"
                )
            path.parent.mkdir(parents=True, exist_ok=True)
            result.to_parquet(path)
            return result

        wrapper.cache_path = lambda *args, **kwargs: _cache_path(  # type: ignore[attr-defined]
            key, signature, args, kwargs, key_args
        )
        wrapper.clear_cache = lambda: _clear_cache(key)  # type: ignore[attr-defined]
        return wrapper  # type: ignore[return-value]

    return decorator


def cache_dir() -> Path:
    """The interim-data directory holding every cached Parquet file."""
    return Path(load_config().paths.data_interim)


def _cache_disabled() -> bool:
    """True when the environment asks for recomputation regardless of cache state."""
    return os.environ.get(CACHE_DISABLED_ENV_VAR, "").strip().lower() in {"1", "true", "yes"}


def _cache_path(
    key: str,
    signature: inspect.Signature,
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
    key_args: Sequence[str] = (),
) -> Path:
    """Build the Parquet path for one call's arguments.

    The fingerprint always covers every argument. ``key_args`` only adds a
    readable label in front of it, so two calls can never share a file merely
    because their labels match.
    """
    parts = [key]
    if key_args:
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        parts.extend(_slug(bound.arguments[name]) for name in key_args)
    parts.append(_fingerprint(signature, args, kwargs))
    return cache_dir() / (_SEPARATOR.join(parts) + CACHE_SUFFIX)


def _slug(value: Any) -> str:
    """Reduce a label argument to a filename-safe fragment."""
    return re.sub(r"[^A-Za-z0-9]+", "-", str(value)).strip("-").lower() or "none"


def _clear_cache(key: str) -> int:
    """Delete every cached file belonging to ``key``. Returns how many were removed."""
    directory = cache_dir()
    if not directory.is_dir():
        return 0
    removed = 0
    for path in directory.glob(f"{key}{_SEPARATOR}*{CACHE_SUFFIX}"):
        path.unlink()
        removed += 1
    return removed


def _fingerprint(
    signature: inspect.Signature,
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
) -> str:
    """Hash the fully-bound call arguments into a short, stable hex digest.

    Defaults are applied first, so calling a function positionally, by keyword, or
    relying on its defaults all map to the same cache entry.
    """
    bound = signature.bind(*args, **kwargs)
    bound.apply_defaults()
    # Sorting by parameter name only — names are unique, so values are never compared.
    payload = ";".join(
        f"{name}={_stable_repr(value, name)}"
        for name, value in sorted(bound.arguments.items())
    )
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=_FINGERPRINT_LENGTH).hexdigest()


def _stable_repr(value: Any, path: str) -> str:
    """Represent an argument reproducibly across processes, or refuse to.

    ``repr`` is not usable directly: dict ordering, object addresses and NumPy
    formatting all vary, which would make the fingerprint unstable or, worse,
    equal for unequal inputs.
    """
    if value is None or isinstance(value, (bool, int, float, str, date, Path)):
        return f"{type(value).__name__}:{value}"
    if isinstance(value, (list, tuple)):
        items = ",".join(_stable_repr(item, f"{path}[{i}]") for i, item in enumerate(value))
        return f"[{items}]"
    if isinstance(value, (set, frozenset)):
        return "{" + ",".join(sorted(_stable_repr(item, path) for item in value)) + "}"
    if isinstance(value, Mapping):
        return (
            "{"
            + ",".join(f"{k}:{_stable_repr(value[k], f'{path}.{k}')}" for k in sorted(value))
            + "}"
        )
    raise TypeError(
        f"@cached cannot fingerprint argument {path!r} of type {type(value).__name__}. "
        "Cached functions take primitives or simple containers of them; pass an "
        "identifier instead of the object itself."
    )
