"""train_patchcore.py end to end on the fake dataset: tiny untrained backbone, file MLflow store.

No download, no Docker. It proves the run gets its metrics, artifacts and model before the
script ever meets the real data.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from anomaly_inspector.data.manifest import build_manifest
from anomaly_inspector.data.splits import assign_splits
from anomaly_inspector.evaluation.metrics import image_auroc

SCRIPT = Path(__file__).parents[2] / "scripts" / "train_patchcore.py"


@pytest.fixture(scope="module")
def script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("train_patchcore", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["train_patchcore"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def project(
    fake_mvtec: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, script: ModuleType
) -> Path:
    """A tiny copy of the repo layout in tmp_path (fake_mvtec already moved there)."""
    manifest = assign_splits(build_manifest(fake_mvtec, ["bottle"]), val_fraction=0.1, seed=42)
    Path("data/processed").mkdir(parents=True)
    manifest.to_parquet("data/processed/manifest.parquet", index=False)
    Path("data/raw/mvtec_ad.dvc").write_text("outs:\n- md5: abc123.dir\n  path: mvtec_ad\n")
    params = {
        "data": {"root": str(fake_mvtec), "categories": ["bottle"], "image_size": 64},
        "patchcore": {"backbone": "resnet18", "coreset_ratio": 0.5},
    }
    Path("params.yaml").write_text(yaml.safe_dump(params))
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@example.com"]
    for command in (["init", "-q"], ["add", "."], ["commit", "-q", "-m", "fake"]):
        subprocess.run([*git, *command], check=True, capture_output=True)

    real = script.build_backbone
    monkeypatch.setattr(script, "build_backbone", lambda name: real(name, pretrained=False))
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    return tmp_path


def the_run() -> Any:
    (run,) = mlflow.search_runs(experiment_names=["patchcore"], output_format="list")
    return run


def test_the_run_holds_the_metrics_the_artifacts_and_a_model_that_reloads(
    script: ModuleType, project: Path
) -> None:
    script.main(["--category", "bottle"])

    run = the_run()
    assert run.info.status == "FINISHED"
    assert run.info.run_name == "patchcore-bottle"
    assert set(run.data.metrics) == {
        "holdout_image_auroc",
        "holdout_pixel_auroc",
        "holdout_aupro",
        "holdout_ms_per_image",
    }
    files = {a.path for a in mlflow.artifacts.list_artifacts(run_id=run.info.run_id)}
    assert files == {"threshold.json", "curves", "gallery"}  # the model is stored apart (MLflow 3)
    nested = {
        a.path
        for folder in ("curves", "gallery")
        for a in mlflow.artifacts.list_artifacts(run_id=run.info.run_id, artifact_path=folder)
    }
    assert nested == {
        "curves/roc.png",
        "curves/pro.png",
        "gallery/failures.jpg",
        "gallery/failures.json",
    }

    (logged_model,) = mlflow.search_logged_models(
        filter_string=f"source_run_id = '{run.info.run_id}'", output_format="list"
    )
    assert logged_model.name == "model"

    threshold = mlflow.artifacts.load_dict(f"runs:/{run.info.run_id}/threshold.json")
    assert threshold["chosen_on"] == "calib"
    assert (threshold["miss_cost"], threshold["false_reject_cost"]) == (10, 1)
    assert isinstance(threshold["threshold"], float)
    assert isinstance(threshold["normal_threshold"], float)
    panels = mlflow.artifacts.load_dict(f"runs:/{run.info.run_id}/gallery/failures.json")
    shown = {p["path"] for p in panels["panels"]}
    manifest = pd.read_parquet("data/processed/manifest.parquet").set_index("path")
    assert set(manifest.loc[sorted(shown), "split"]) == {"holdout"}
    # Each picture is the image its score belongs to: a "missed" panel shows a defect.
    assert all((p["kind"] == "missed") == (p["defect_type"] != "good") for p in panels["panels"])

    # The registered model, loaded fresh from the store, gives back the logged holdout AUROC.
    version = mlflow.MlflowClient().get_model_version("patchcore-bottle", "1")
    assert version.run_id == run.info.run_id
    model = mlflow.pytorch.load_model("models:/patchcore-bottle/1")
    holdout = manifest[manifest["split"] == "holdout"].reset_index()
    reloaded = script.score(model, holdout, 64).scores
    assert image_auroc(holdout["label"].to_numpy(), reloaded) == pytest.approx(
        run.data.metrics["holdout_image_auroc"]
    )


def test_the_threshold_is_chosen_on_calib_and_val_normal_never_on_holdout(
    script: ModuleType, project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scored: dict[str, Any] = {}  # split -> the scores the script computed on it
    seen: dict[str, Any] = {}  # threshold function -> the scores it was given
    real_score, real_cost, real_normal = (
        script.score,
        script.cost_threshold,
        script.normal_threshold,
    )

    def spy_score(model: torch.nn.Module, rows: pd.DataFrame, size: int) -> Any:
        predictions = real_score(model, rows, size)
        (split,) = set(rows["split"])
        scored[split] = predictions.scores
        return predictions

    def spy_cost(labels: Any, scores: Any, *costs: float) -> float:
        seen["cost"] = (labels, scores)
        return float(real_cost(labels, scores, *costs))

    def spy_normal(scores: Any, quantile: float) -> float:
        seen["normal"] = scores
        return float(real_normal(scores, quantile))

    monkeypatch.setattr(script, "score", spy_score)
    monkeypatch.setattr(script, "cost_threshold", spy_cost)
    monkeypatch.setattr(script, "normal_threshold", spy_normal)
    script.main(["--category", "bottle"])

    calib = pd.read_parquet("data/processed/manifest.parquet").query("split == 'calib'")
    assert set(scored) == {"holdout", "calib", "val_normal"}
    labels, scores = seen["cost"]
    np.testing.assert_array_equal(labels, calib["label"].to_numpy())
    np.testing.assert_array_equal(scores, scored["calib"])
    np.testing.assert_array_equal(seen["normal"], scored["val_normal"])
