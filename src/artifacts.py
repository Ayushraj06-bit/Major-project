"""Run artifact store.

Everything an experiment produces — predictions, metrics, SHAP values, the
serialised config, model weights — is written through :func:`save_run` into
``results/runs/<name>/`` and read back through :func:`load_run`.

The dashboard reads from here and computes nothing itself. That is the point of
the module: if a number appears on screen it was produced by a recorded run and
can be traced back to the configuration that produced it, rather than being
recalculated on the fly by display code.

Each run directory carries a ``manifest.json`` recording what was stored and how,
so :func:`load_run` reconstructs objects from a declaration rather than guessing
from file extensions.
"""

from __future__ import annotations

import json
import pickle
import re
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import load_config

MANIFEST_FILENAME = "manifest.json"
MANIFEST_VERSION = 1

_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

_JSON_TYPES = (dict, list, tuple, str, int, float, bool)


class ArtifactError(RuntimeError):
    """Raised when a run cannot be written or read back faithfully."""


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def save_run(name: str, *, overwrite: bool = False, **objects: Any) -> Path:
    """Write every keyword object into ``results/runs/<name>/``.

    The storage format is chosen from each object's type: DataFrames become
    Parquet, arrays ``.npy``, JSON-compatible values ``.json``, objects exposing
    ``.save()`` (such as a Keras model) delegate to it, and anything else is
    pickled.

    Args:
        name: Run identifier, also the directory name.
        overwrite: Replace an existing run of this name. Off by default, so a
            re-run cannot quietly destroy results already cited in the report.
        **objects: Named artifacts, e.g. ``predictions=df, metrics={...}``.

    Returns:
        The run directory.

    Raises:
        ValueError: ``name`` is unsafe as a directory name, or no objects given.
        FileExistsError: the run exists and ``overwrite`` is False.
        ArtifactError: an object could not be written.
    """
    if not objects:
        raise ValueError(f"save_run({name!r}): nothing to save")

    directory = run_dir(name)
    if directory.exists():
        if not overwrite:
            raise FileExistsError(
                f"run {name!r} already exists at {directory}. "
                "Pass overwrite=True to replace it, or choose another name."
            )
        _clear_directory(directory)
    directory.mkdir(parents=True, exist_ok=True)

    entries: dict[str, dict[str, str]] = {}
    for label, obj in objects.items():
        kind = _classify(obj)
        suffix, write = _WRITERS[kind]
        filename = f"{label}{suffix}"
        try:
            write(obj, directory / filename)
        except Exception as exc:  # noqa: BLE001 - re-raised with the artifact's name attached
            raise ArtifactError(
                f"run {name!r}: could not write artifact {label!r} as {kind}: {exc}"
            ) from exc
        entries[label] = {"kind": kind, "filename": filename}

    manifest = {
        "version": MANIFEST_VERSION,
        "name": name,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "entries": entries,
    }
    (directory / MANIFEST_FILENAME).write_text(
        json.dumps(manifest, indent=2, default=_json_default), encoding="utf-8"
    )
    return directory


def load_run(name: str) -> dict[str, Any]:
    """Read back every artifact of a saved run, keyed by the label it was saved under.

    Objects stored via a ``.save()`` method are returned as the :class:`Path` they
    were written to, because this module deliberately does not import a modelling
    framework — the caller that knows the framework loads them.

    Raises:
        FileNotFoundError: no such run, or its manifest is missing.
        ArtifactError: the manifest is unreadable or references a missing file.
    """
    directory = run_dir(name)
    manifest_path = directory / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"run {name!r} has no {MANIFEST_FILENAME} at {directory}")

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries: dict[str, dict[str, str]] = manifest["entries"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ArtifactError(f"run {name!r}: malformed manifest at {manifest_path}: {exc}") from exc

    loaded: dict[str, Any] = {}
    for label, entry in entries.items():
        path = directory / entry["filename"]
        if not path.exists():
            raise ArtifactError(f"run {name!r}: manifest lists {label!r} but {path} is missing")
        read = _READERS.get(entry["kind"])
        if read is None:
            raise ArtifactError(
                f"run {name!r}: artifact {label!r} has unknown kind {entry['kind']!r}"
            )
        loaded[label] = read(path)
    return loaded


def run_dir(name: str) -> Path:
    """The directory a run is stored in, without creating it.

    Raises:
        ValueError: ``name`` contains path separators or traversal segments.
    """
    if not _NAME_PATTERN.match(name):
        raise ValueError(
            f"run name {name!r} must match {_NAME_PATTERN.pattern} — no separators or traversal"
        )
    return Path(load_config().paths.runs) / name


def list_runs() -> list[str]:
    """Names of every complete run on disk, sorted.

    A run is complete when it has a manifest; a half-written directory is not
    listed. Nothing calls this yet -- the dashboard reads one fixed run by name --
    so it exists for inspection rather than for a selector.
    """
    root = Path(load_config().paths.runs)
    if not root.is_dir():
        return []
    return sorted(
        path.name for path in root.iterdir() if (path / MANIFEST_FILENAME).is_file()
    )


# --------------------------------------------------------------------------- #
# Format dispatch — one table, so writers and readers cannot drift apart
# --------------------------------------------------------------------------- #


def _classify(obj: Any) -> str:
    """Pick the storage format for one object. Order matters: narrow types first."""
    if isinstance(obj, pd.DataFrame):
        return "dataframe"
    if isinstance(obj, pd.Series):
        return "series"
    if isinstance(obj, np.ndarray):
        return "array"
    if obj is None or isinstance(obj, _JSON_TYPES):
        return "json"
    if callable(getattr(obj, "save", None)):
        return "model"
    return "pickle"


def _write_json(obj: Any, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, default=_json_default), encoding="utf-8")


def _write_pickle(obj: Any, path: Path) -> None:
    with path.open("wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _read_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


_WRITERS: dict[str, tuple[str, Callable[[Any, Path], None]]] = {
    "dataframe": (".parquet", lambda obj, path: obj.to_parquet(path)),
    "series": (
        ".parquet",
        lambda obj, path: obj.to_frame(name=obj.name or "value").to_parquet(path),
    ),
    "array": (".npy", lambda obj, path: np.save(path, obj)),
    "json": (".json", _write_json),
    "model": (".keras", lambda obj, path: obj.save(path)),
    "pickle": (".pkl", _write_pickle),
}

_READERS: dict[str, Callable[[Path], Any]] = {
    "dataframe": pd.read_parquet,
    "series": lambda path: pd.read_parquet(path).iloc[:, 0],
    "array": np.load,
    "json": lambda path: json.loads(path.read_text(encoding="utf-8")),
    # Returned as a path: loading it needs the framework that wrote it.
    "model": lambda path: path,
    "pickle": _read_pickle,
}


def _json_default(obj: Any) -> Any:
    """Serialise the few non-JSON types that reach metrics dicts."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    raise TypeError(f"cannot serialise {type(obj).__name__} to JSON")


def _clear_directory(directory: Path) -> None:
    """Empty a run directory, keeping the directory itself.

    Deliberately does not remove the directory. On Windows a synced folder
    (OneDrive, Dropbox) frequently holds a handle on it, so ``rmdir`` raises
    PermissionError even once the contents are gone. Reusing the directory has
    the same effect for callers and cannot fail that way.
    """
    for path in sorted(directory.rglob("*"), reverse=True):
        if path.is_dir():
            path.rmdir()
        else:
            path.unlink()
