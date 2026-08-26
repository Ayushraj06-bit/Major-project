"""The notebooks stay thin.

brain.md section 7 rules that notebooks define no functions and import from
``src/``. That rule is worth enforcing rather than trusting: logic written in a
cell cannot be tested, and it drifts from the module it was copied from, which is
how a report's figures stop matching the code that ships.

These check structure only. Executing them means training an LSTM and running
SHAP, which belongs in a manual run rather than in a suite people need to run
often.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS = sorted((PROJECT_ROOT / "notebooks").glob("*.ipynb"))

#: The four the README's repository structure calls for.
REQUIRED = {
    "01_data_exploration.ipynb",
    "02_feature_engineering.ipynb",
    "03_model_training.ipynb",
    "04_shap_analysis.ipynb",
}


def _code(path: Path) -> str:
    """Every code cell of a notebook, concatenated."""
    cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
    sources = []
    for cell in cells:
        if cell["cell_type"] != "code":
            continue
        source = cell["source"]
        sources.append(source if isinstance(source, str) else "".join(source))
    return "\n".join(sources)


def test_the_required_notebooks_exist() -> None:
    """README section 12 names four; a missing one is an incomplete deliverable."""
    assert {path.name for path in NOTEBOOKS} >= REQUIRED


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_is_valid_json_and_parses(path: Path) -> None:
    """A notebook that will not parse is worse than no notebook."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["cells"], f"{path.name} has no cells"
    ast.parse(_code(path))


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_defines_no_functions_or_classes(path: Path) -> None:
    """The rule: notebooks call into src, they do not reimplement it."""
    tree = ast.parse(_code(path))
    defined = [
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
    ]
    assert not defined, (
        f"{path.name} defines {defined}; that logic belongs in src/ where it can "
        "be tested"
    )


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_imports_from_src(path: Path) -> None:
    """A notebook that imports nothing from the project is not exercising it."""
    tree = ast.parse(_code(path))
    modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert any(module.startswith("src.") for module in modules), (
        f"{path.name} imports nothing from src/"
    )


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_has_no_stored_outputs(path: Path) -> None:
    """Committed outputs go stale and make diffs unreadable.

    They also invite reading a number that was produced by code the notebook no
    longer contains.
    """
    cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
    with_output = [
        position
        for position, cell in enumerate(cells)
        if cell["cell_type"] == "code" and cell.get("outputs")
    ]
    assert not with_output, f"{path.name} has stored output in cells {with_output}"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_states_the_synthetic_caveat(path: Path) -> None:
    """Every notebook must say its numbers may describe a generator, not dengue."""
    text = path.read_text(encoding="utf-8").lower()
    assert "synthetic" in text or "generator" in text, (
        f"{path.name} does not warn that its figures may not be real"
    )
