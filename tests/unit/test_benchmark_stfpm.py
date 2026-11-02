"""The benchmark script's STFPM: trained and scored like the PatchCores, reproducible, logged.

On the fake project with a tiny random ResNet-18 and 2 epochs, so it runs with the fast tests
(the anomalib ones are slow and need the bench extra).
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import mlflow
import pytest

SCRIPT = Path(__file__).parents[2] / "scripts" / "benchmark_anomalib.py"
METRICS = ["image_auroc", "pixel_auroc", "aupro"]


def run_stfpm(tracking_uri: str | None = None) -> dict[str, Any]:
    env = {k: v for k, v in os.environ.items() if k != "MLFLOW_TRACKING_URI"}
    if tracking_uri:
        env["MLFLOW_TRACKING_URI"] = tracking_uri
    command = [
        sys.executable,
        str(SCRIPT),
        "--category=bottle",
        "--impl=stfpm",
        "--out=results.json",
        "--no-pretrained",
        "--size=64",
        "--batch-size=2",
        "--epochs=2",
    ]
    done = subprocess.run(command, capture_output=True, text=True, check=False, env=env)
    assert done.returncode == 0, done.stderr[-2000:]
    assert "epoch 1: train_loss" in done.stdout  # each epoch is reported as it ends
    entry: dict[str, Any] = json.loads(Path("results.json").read_text())["bottle/stfpm"]
    return entry


def test_stfpm_is_scored_on_the_same_images_and_a_rerun_gives_the_same_numbers(
    fake_project: Path,
) -> None:
    first = run_stfpm()

    assert all(0.0 <= first[metric] <= 1.0 for metric in METRICS)
    # The PatchCores' training and test images, plus val_normal for early stopping.
    assert (first["train_images"], first["val_images"], first["test_images"]) == (3, 1, 4)
    assert first["epochs_run"] == 2
    assert 0 <= first["best_epoch"] < 2
    assert first["settings"]["backbone"] == "resnet18"  # STFPM's own default, not PatchCore's
    assert first["settings"]["stfpm"]["epochs"] == 2  # the command line wins over params.yaml
    assert "coreset_ratio" not in first["settings"]

    second = run_stfpm()  # the same seed: the same student, batches and scores
    for key in [*METRICS, "best_val_loss", "best_epoch"]:
        assert second[key] == first[key]


def test_with_a_tracking_server_the_run_and_every_epoch_are_logged(fake_project: Path) -> None:
    uri = f"sqlite:///{fake_project / 'mlflow.db'}"
    entry = run_stfpm(tracking_uri=uri)

    client = mlflow.MlflowClient(uri)
    experiment = client.get_experiment_by_name("stfpm")
    assert experiment is not None
    (run,) = client.search_runs([experiment.experiment_id])
    assert run.info.status == "FINISHED"
    assert run.data.tags["category"] == "bottle"
    assert run.data.tags["evaluated_on"] == "calib + holdout"
    assert run.data.params["stfpm.epochs"] == "2"
    assert len(client.get_metric_history(run.info.run_id, "val_loss")) == 2
    assert run.data.metrics["test_aupro"] == pytest.approx(entry["aupro"])
    assert run.data.metrics["best_epoch"] == entry["best_epoch"]
