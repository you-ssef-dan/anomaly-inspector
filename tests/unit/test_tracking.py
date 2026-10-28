"""tracking.py: when a run counts as dirty, file fingerprints, and the tracking switch."""

import hashlib
import subprocess
import time
from pathlib import Path

import mlflow
import pytest

from anomaly_inspector.config import DataConfig, Params
from anomaly_inspector.tracking import (
    git_is_dirty,
    git_sha,
    md5_of,
    start_run,
    tracking_enabled,
    tracking_uri,
)

GIT = ["git", "-c", "user.name=t", "-c", "user.email=t@example.com"]


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A committed repo shaped like ours: code, params, a pipeline lock file and reports."""
    monkeypatch.chdir(tmp_path)
    for path in (
        "src/code.py",
        "params.yaml",
        "dvc.yaml",
        "dvc.lock",
        "reports/bottle/metrics.json",
    ):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("first\n")
    Path("data/raw").mkdir(parents=True)
    Path("data/raw/mvtec_ad.dvc").write_text("outs:\n- md5: abc123.dir\n  path: mvtec_ad\n")
    for command in (["init", "-q"], ["add", "."], ["commit", "-q", "-m", "first"]):
        subprocess.run([*GIT, *command], check=True, capture_output=True)
    return tmp_path


def test_a_clean_repo_is_not_dirty(repo: Path) -> None:
    assert not git_is_dirty()


@pytest.mark.parametrize("path", ["src/code.py", "params.yaml", "dvc.yaml"])
def test_an_uncommitted_change_to_a_source_file_is_dirty(repo: Path, path: str) -> None:
    Path(path).write_text("changed\n")
    assert git_is_dirty()


def test_what_dvc_repro_writes_does_not_make_a_run_dirty(repo: Path) -> None:
    # dvc repro rewrites these while the pipeline runs; they are results, not inputs.
    Path("dvc.lock").write_text("changed\n")
    Path("reports/bottle/metrics.json").write_text("changed\n")
    assert not git_is_dirty()


def test_a_new_untracked_file_does_not_make_a_run_dirty(repo: Path) -> None:
    Path("notes.txt").write_text("scratch\n")
    assert not git_is_dirty()


def test_the_md5_is_the_one_of_the_file_bytes(tmp_path: Path) -> None:
    content = bytes(range(256)) * 9000  # bigger than one read block
    (tmp_path / "model.pt").write_bytes(content)
    assert md5_of(tmp_path / "model.pt") == hashlib.md5(content).hexdigest()


@pytest.mark.parametrize(("value", "enabled"), [("http://localhost:5000", True), ("", False)])
def test_tracking_is_on_only_with_a_server_address(
    value: str, enabled: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", value)
    assert tracking_enabled() is enabled


def test_tracking_is_off_without_the_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    assert not tracking_enabled()
    with pytest.raises(RuntimeError, match="MLFLOW_TRACKING_URI"):
        tracking_uri()


def test_the_uri_comes_back_without_surrounding_spaces(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "  http://localhost:5000 ")
    assert tracking_uri() == "http://localhost:5000"


def test_a_run_records_its_origin_its_params_and_the_machine_load(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{repo / 'store.db'}")
    # Sample every 0.2 s instead of every 10 s, so a one-second run gets samples.
    monkeypatch.setenv("MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL", "0.2")
    params = Params(data=DataConfig(root=Path("data/raw/mvtec_ad"), categories=["bottle"]))

    with start_run("patchcore", "patchcore-bottle", "bottle", params) as run:
        time.sleep(1)

    data = mlflow.get_run(run.info.run_id).data
    assert data.tags["git_sha"] == git_sha()
    assert data.tags["git_dirty"] == "False"
    assert data.tags["data_md5"] == "abc123.dir"
    assert data.tags["category"] == "bottle"
    assert data.params["stfpm.lr"] == "0.4"
    assert data.params["patchcore.coreset_ratio"] == "0.01"
    assert {"system/cpu_utilization_percentage", "system/system_memory_usage_megabytes"} <= set(
        data.metrics
    )
