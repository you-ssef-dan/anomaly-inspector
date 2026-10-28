"""backfill_experiments.py: Phase 2 results become MLflow runs, once each, traced to a commit."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import mlflow
import pytest

SCRIPT = Path(__file__).parents[2] / "scripts" / "backfill_experiments.py"
GIT = ["git", "-c", "user.name=t", "-c", "user.email=t@example.com"]


def result(aupro: float, **settings: Any) -> dict[str, Any]:
    """One entry shaped like reports/experiments.json."""
    return {
        "aupro": aupro,
        "image_auroc": 0.9,
        "pixel_auroc": 0.98,
        "ms_per_image": 150.0,
        "fit_seconds": 60.0,
        "peak_ram_mb": 2700.0,
        "memory_bank_mb": 9.4,
        "memory_bank_rows": 1536,
        "calib_images": 41,
        "train_images": 150,
        "coreset_ratio_used": 0.01,
        "settings": {
            "backbone": "wide_resnet50_2",
            "coreset_ratio": 0.01,
            "images": "all",
            "size": 256,
            "subset_seed": None,
        }
        | settings,
    }


@pytest.fixture(scope="module")
def script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("backfill_experiments", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["backfill_experiments"] = module
    spec.loader.exec_module(module)
    return module


def commit_results(results: dict[str, Any]) -> str:
    Path("reports/experiments.json").write_text(json.dumps(results))
    subprocess.run([*GIT, "add", "."], check=True, capture_output=True)
    subprocess.run([*GIT, "commit", "-q", "-m", "results"], check=True, capture_output=True)
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A git repo holding a DVC pointer file, with a file MLflow store next to it."""
    monkeypatch.chdir(tmp_path)
    Path("reports").mkdir()
    Path("data/raw").mkdir(parents=True)
    Path("data/raw/mvtec_ad.dvc").write_text("outs:\n- md5: abc123.dir\n  path: mvtec_ad\n")
    subprocess.run([*GIT, "init", "-q"], check=True, capture_output=True)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    return tmp_path


def backfilled() -> dict[str, Any]:
    runs = mlflow.search_runs(experiment_names=["phase2-experiments"], output_format="list")
    by_name = {run.info.run_name: run for run in runs}
    assert len(by_name) == len(runs), "the same result was logged twice"
    return by_name


def test_each_result_becomes_a_run_traced_to_the_commit_that_holds_it(
    script: ModuleType, repo: Path
) -> None:
    commit = commit_results(
        {
            "bottle/baseline": result(0.94),
            "screw/images/5/seed1": result(0.70, images=5, subset_seed=1),
        }
    )
    # A later commit that does not touch the results (but changes the data) must not become
    # their source: the runs keep the code and data of the commit that holds the results.
    Path("data/raw/mvtec_ad.dvc").write_text("outs:\n- md5: def456.dir\n  path: mvtec_ad\n")
    subprocess.run([*GIT, "add", "."], check=True, capture_output=True)
    subprocess.run([*GIT, "commit", "-q", "-m", "later"], check=True, capture_output=True)

    script.main([])

    runs = backfilled()
    assert set(runs) == {"bottle/baseline", "screw/images/5/seed1"}
    screw = runs["screw/images/5/seed1"].data
    assert screw.tags["category"] == "screw"
    assert screw.tags["experiment"] == "images"
    assert screw.tags["git_sha"] == commit
    assert screw.tags["data_md5"] == "abc123.dir"
    assert screw.tags["backfilled_from"] == "reports/experiments.json"
    assert "git_dirty" not in screw.tags  # unknown for old results, so never claimed
    assert screw.params["images"] == "5"
    assert screw.params["subset_seed"] == "1"
    assert screw.metrics["calib_aupro"] == 0.70  # scored on calib, and named so
    assert "aupro" not in screw.metrics
    assert screw.metrics["ms_per_image"] == 150.0
    assert "subset_seed" not in runs["bottle/baseline"].data.params  # None is not a setting


def test_running_it_again_adds_only_the_new_results(script: ModuleType, repo: Path) -> None:
    results = {"bottle/baseline": result(0.94)}
    commit_results(results)
    script.main([])
    script.main([])
    assert len(backfilled()) == 1

    commit_results(results | {"bottle/size/320": result(0.95, size=320)})
    script.main([])
    assert set(backfilled()) == {"bottle/baseline", "bottle/size/320"}


def test_uncommitted_results_are_refused(script: ModuleType, repo: Path) -> None:
    Path("reports/experiments.json").write_text(json.dumps({"bottle/baseline": result(0.94)}))
    subprocess.run([*GIT, "add", "data"], check=True, capture_output=True)
    subprocess.run([*GIT, "commit", "-q", "-m", "data only"], check=True, capture_output=True)
    with pytest.raises(ValueError, match="not committed"):
        script.main([])


def test_without_a_server_nothing_is_written(
    script: ModuleType, repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commit_results({"bottle/baseline": result(0.94)})
    monkeypatch.delenv("MLFLOW_TRACKING_URI")
    with pytest.raises(RuntimeError, match="MLFLOW_TRACKING_URI"):
        script.main([])
    assert not (repo / "mlruns").exists()
    assert not (repo / "mlflow.db").exists()  # the throwaway store was never created either
