"""dvc.yaml against the code: every stage depends on exactly the modules it runs.

Too narrow, and DVC skips a stage whose code changed, so its results go stale without a word.
Too wide, and a docstring edit elsewhere re-runs it. Both are caught here instead.
"""

import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from anomaly_inspector.pipelines.evaluate_patchcore import Outputs
from anomaly_inspector.pipelines.train_patchcore import model_path

REPO = Path(__file__).parents[2]
# The package's own __init__.py holds only the "hello" command uv generated; no stage runs it.
NOT_STAGE_CODE = {"src/anomaly_inspector/__init__.py"}
STAGES: dict[str, Any] = yaml.safe_load((REPO / "dvc.yaml").read_text())["stages"]
CATEGORY = "bottle"  # foreach stages are checked for one category: they share one definition


def definition(stage: dict[str, Any]) -> dict[str, Any]:
    body = stage.get("do", stage)
    return yaml.safe_load(yaml.safe_dump(body).replace("${item}", CATEGORY))  # type: ignore[no-any-return]


def paths(entries: list[Any]) -> set[str]:
    """dvc.yaml lists a path either as a string or as {path: options}."""
    return {entry if isinstance(entry, str) else next(iter(entry)) for entry in entries}


def imported_files(module: str) -> set[str]:
    """The project's own source files that importing the stage's module loads (empty
    __init__.py files left out), found in a fresh interpreter."""
    code = (
        "import importlib, sys\n"
        f"importlib.import_module({module!r})\n"
        "for name, mod in sys.modules.items():\n"
        "    if name.startswith('anomaly_inspector') and mod.__file__:\n"
        "        print(mod.__file__)\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO, check=True, capture_output=True, text=True
    ).stdout
    files = {Path(line).resolve() for line in out.split()}
    return {
        str(f.relative_to(REPO))
        for f in files
        if f.is_relative_to(REPO / "src")
        and not (f.name == "__init__.py" and f.stat().st_size == 0)
    } - NOT_STAGE_CODE


@pytest.mark.parametrize("name", sorted(STAGES))
def test_the_stage_depends_on_exactly_the_code_it_imports(name: str) -> None:
    stage = definition(STAGES[name])
    module = stage["cmd"].split("-m ")[1].split()[0]
    imported = imported_files(module)
    code_deps = {dep.rstrip("/") for dep in paths(stage["deps"]) if dep.startswith("src/")}

    def covered(file: str) -> bool:
        return any(file == dep or file.startswith(dep + "/") for dep in code_deps)

    assert {f for f in imported if not covered(f)} == set(), "imported but not in deps"
    unused = {
        dep for dep in code_deps if not any(f == dep or f.startswith(dep + "/") for f in imported)
    }
    assert unused == set(), "in deps but never imported"


def test_the_stage_outputs_are_where_the_code_writes() -> None:
    train = definition(STAGES["train_patchcore"])
    assert paths(train["outs"]) == {str(model_path(CATEGORY))}

    evaluate = definition(STAGES["evaluate_patchcore"])
    out = Outputs.for_category(CATEGORY)
    assert paths(evaluate["outs"]) == {str(out.threshold), str(out.figures)}
    assert paths(evaluate["metrics"]) == {str(out.metrics)}
    assert paths(evaluate["plots"]) == {str(out.roc), str(out.pro)}
    assert str(model_path(CATEGORY)) in paths(evaluate["deps"])  # it scores what train saved


def test_every_category_gets_its_stages() -> None:
    params = yaml.safe_load((REPO / "params.yaml").read_text())
    for name in ("train_patchcore", "evaluate_patchcore"):
        assert STAGES[name]["foreach"] == "${data.categories}"
    assert params["data"]["categories"] == ["bottle", "carpet", "screw"]
