"""The benchmark script end to end on the fake dataset: a tiny random backbone, no download.

Run with `make test-slow`. Skipped when anomalib is not installed, which is how CI runs.
"""

import json
import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path

import pytest

from anomaly_inspector.data.manifest import build_manifest
from anomaly_inspector.data.splits import assign_splits

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(find_spec("anomalib") is None, reason="needs the bench extra"),
]
SCRIPT = Path(__file__).parents[2] / "scripts" / "benchmark_anomalib.py"
METRICS = ["image_auroc", "pixel_auroc", "aupro"]


@pytest.fixture
def manifest_path(fake_mvtec: Path) -> Path:
    manifest = assign_splits(build_manifest(fake_mvtec, ["bottle"]), val_fraction=0.1, seed=42)
    path = Path("manifest.parquet")
    manifest.to_parquet(path, index=False)
    return path


def run_benchmark(*extra: str) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        str(SCRIPT),
        "--category=bottle",
        "--manifest=manifest.parquet",
        "--out=results.json",
        "--backbone=resnet18",
        "--no-pretrained",
        "--size=64",
        "--batch-size=2",
        "--coreset-ratio=0.5",
        *extra,
    ]
    return subprocess.run(command, capture_output=True, text=True, check=False)


def test_each_implementation_writes_its_result_into_one_file(manifest_path: Path) -> None:
    for extra in (
        ["--impl=ours"],
        ["--impl=anomalib-v1"],
        ["--impl=anomalib-v1", "--num-neighbors=1"],
    ):
        result = run_benchmark(*extra)
        assert result.returncode == 0, result.stderr[-2000:]

    results = json.loads(Path("results.json").read_text())
    assert set(results) == {"bottle/ours", "bottle/anomalib-v1-k9", "bottle/anomalib-v1-k1"}
    for entry in results.values():
        assert all(0.0 <= entry[metric] <= 1.0 for metric in METRICS)
        assert entry["train_images"] == 3
        assert entry["test_images"] == 4
        assert entry["settings"]["image_size"] == 64
    assert results["bottle/ours"]["settings"]["num_neighbors"] is None
    assert results["bottle/anomalib-v1-k1"]["settings"]["num_neighbors"] == 1


def test_anomalib_with_our_weights_is_proven_to_see_the_same_features(manifest_path: Path) -> None:
    result = run_benchmark("--impl=anomalib-v1")
    assert result.returncode == 0, result.stderr[-2000:]
    assert "features are identical" in result.stdout


def test_a_second_run_replaces_its_own_entry_and_keeps_the_others(manifest_path: Path) -> None:
    for _ in range(2):
        assert run_benchmark("--impl=ours").returncode == 0
    assert run_benchmark("--impl=anomalib-v1").returncode == 0
    assert set(json.loads(Path("results.json").read_text())) == {
        "bottle/ours",
        "bottle/anomalib-v1-k9",
    }


def test_an_unknown_implementation_is_refused(manifest_path: Path) -> None:
    result = run_benchmark("--impl=tensorflow")
    assert result.returncode != 0
    assert "invalid choice" in result.stderr
