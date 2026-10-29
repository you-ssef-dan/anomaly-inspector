"""The DVC stages train_patchcore and evaluate_patchcore, run on the fake dataset."""

import json
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from torch.utils.data import DataLoader

from anomaly_inspector.config import load_params
from anomaly_inspector.data.dataset import ManifestDataset
from anomaly_inspector.evaluation import thresholds
from anomaly_inspector.evaluation.evaluate import evaluate, load_masks, predict
from anomaly_inspector.export.onnx_export import OnnxPatchCore, read_metadata
from anomaly_inspector.models.storage import load_patchcore
from anomaly_inspector.pipelines import (
    evaluate_patchcore,
    export_patchcore,
    optimize_patchcore,
    train_patchcore,
)
from anomaly_inspector.pipelines.evaluate_patchcore import Outputs
from anomaly_inspector.tracking import git_sha, md5_of

MODEL = Path("models/patchcore/bottle/model.pt")


def manifest() -> pd.DataFrame:
    """The bottle rows of the fake project's manifest."""
    rows = pd.read_parquet("data/processed/manifest.parquet")
    return rows[rows["category"] == "bottle"]


def test_train_saves_the_model_it_fitted(fake_project: Path) -> None:
    train_patchcore.main(["--category", "bottle"])

    saved = load_patchcore(MODEL)
    assert (saved.backbone, saved.image_size) == ("resnet18", 64)
    rows = manifest()
    fresh = train_patchcore.train_patchcore(load_params(), rows[rows["split"] == "train"])
    assert torch.equal(saved.model.memory_bank, fresh.memory_bank)


def test_training_twice_gives_the_same_file(fake_project: Path) -> None:
    train_patchcore.main(["--category", "bottle"])
    first = MODEL.read_bytes()
    train_patchcore.main(["--category", "bottle"])
    assert MODEL.read_bytes() == first


def test_train_reads_only_the_train_split(
    fake_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[pd.DataFrame] = []
    real = train_patchcore.train_patchcore

    def spy(params: Any, rows: pd.DataFrame) -> Any:
        seen.append(rows)
        return real(params, rows)

    monkeypatch.setattr(train_patchcore, "train_patchcore", spy)
    train_patchcore.main(["--category", "bottle"])
    (rows,) = seen
    assert set(rows["split"]) == {"train"}
    assert set(rows["category"]) == {"bottle"}  # the fake project also holds screw


def test_an_unknown_category_is_refused_before_any_work(fake_project: Path) -> None:
    with pytest.raises(ValueError, match="no train images"):
        train_patchcore.main(["--category", "screwdriver"])
    assert not Path("models").exists()


def test_evaluate_writes_every_output_dvc_expects(fake_project: Path) -> None:
    train_patchcore.main(["--category", "bottle"])
    evaluate_patchcore.main(["--category", "bottle"])

    out = Outputs.for_category("bottle")
    metrics = json.loads(out.metrics.read_text())
    assert set(metrics) == {
        "holdout_image_auroc",
        "holdout_pixel_auroc",
        "holdout_aupro",
        "holdout_ms_per_image",
    }
    assert all(0 <= metrics[f"holdout_{m}"] <= 1 for m in ("image_auroc", "pixel_auroc", "aupro"))

    threshold = json.loads(out.threshold.read_text())
    assert threshold["chosen_on"] == "calib"
    assert (threshold["miss_cost"], threshold["false_reject_cost"]) == (10, 1)

    roc = pd.read_csv(out.roc)
    assert list(roc.columns) == ["fpr", "tpr"]
    assert tuple(roc.iloc[0]) == (0, 0)
    assert tuple(roc.iloc[-1]) == (1, 1)
    pro = pd.read_csv(out.pro)
    assert list(pro.columns) == ["fpr", "pro"]
    assert len(pro) == 1002

    assert {p.name for p in out.figures.iterdir()} == {
        "roc.png",
        "pro.png",
        "failures.jpg",
        "failures.json",
    }
    panels = json.loads((out.figures / "failures.json").read_text())["panels"]
    shown = manifest().set_index("path").loc[sorted({p["path"] for p in panels}), "split"]
    assert set(shown) == {"holdout"}


def test_evaluate_scores_at_the_size_the_model_was_trained_at(
    fake_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    train_patchcore.main(["--category", "bottle"])
    # params.yaml changes after training: the saved model still expects 64 px.
    params = yaml.safe_load(Path("params.yaml").read_text())
    params["data"]["image_size"] = 96
    Path("params.yaml").write_text(yaml.safe_dump(params))
    sizes: list[int] = []
    real = evaluate_patchcore.score

    def spy(model: Any, rows: pd.DataFrame, size: int) -> Any:
        sizes.append(size)
        return real(model, rows, size)

    monkeypatch.setattr(evaluate_patchcore, "score", spy)
    evaluate_patchcore.main(["--category", "bottle"])
    assert set(sizes) == {64}


def test_the_threshold_is_chosen_on_calib_and_val_normal_never_on_holdout(
    fake_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    train_patchcore.main(["--category", "bottle"])
    scored: dict[str, Any] = {}  # split -> the scores computed on it
    seen: dict[str, Any] = {}  # threshold function -> what it was given
    real_score = evaluate_patchcore.score

    def spy_score(model: Any, rows: pd.DataFrame, size: int) -> Any:
        predictions = real_score(model, rows, size)
        (split,) = set(rows["split"])
        scored[split] = predictions.scores
        return predictions

    def spy_cost(labels: Any, scores: Any, *costs: float) -> float:
        seen["cost"] = (labels, scores)
        return thresholds.cost_threshold(labels, scores, *costs)

    def spy_normal(scores: Any, quantile: float) -> float:
        seen["normal"] = scores
        return thresholds.normal_threshold(scores, quantile)

    monkeypatch.setattr(evaluate_patchcore, "score", spy_score)
    # Patched where the stage looks them up, so the stage calls the spies.
    monkeypatch.setattr(evaluate_patchcore, "cost_threshold", spy_cost)
    monkeypatch.setattr(evaluate_patchcore, "normal_threshold", spy_normal)
    evaluate_patchcore.main(["--category", "bottle"])

    calib = manifest().query("split == 'calib'")
    assert set(scored) == {"holdout", "calib", "val_normal"}
    labels, scores = seen["cost"]
    np.testing.assert_array_equal(labels, calib["label"].to_numpy())
    np.testing.assert_array_equal(scores, scored["calib"])
    np.testing.assert_array_equal(seen["normal"], scored["val_normal"])


def test_evaluate_leaves_the_model_untouched(fake_project: Path) -> None:
    train_patchcore.main(["--category", "bottle"])
    before = MODEL.read_bytes()
    evaluate_patchcore.main(["--category", "bottle"])
    assert MODEL.read_bytes() == before


def test_without_a_server_the_stage_writes_only_its_files(
    fake_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    train_patchcore.main(["--category", "bottle"])
    evaluate_patchcore.main(["--category", "bottle"])
    assert Outputs.for_category("bottle").metrics.exists()
    assert not Path("mlflow.db").exists()
    assert not Path("mlruns").exists()


def test_with_a_server_the_evaluation_is_a_run_tagged_with_its_model(
    fake_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{fake_project / 'store.db'}")
    train_patchcore.main(["--category", "bottle"])
    evaluate_patchcore.main(["--category", "bottle"])

    (run,) = mlflow.search_runs(experiment_names=["patchcore"], output_format="list")
    assert (run.info.run_name, run.info.status) == ("patchcore-bottle", "FINISHED")
    assert run.data.tags["model_md5"] == md5_of(MODEL)
    assert run.data.tags["category"] == "bottle"
    out = Outputs.for_category("bottle")
    logged = {k: v for k, v in run.data.metrics.items() if not k.startswith("system/")}
    assert logged == pytest.approx(json.loads(out.metrics.read_text()))
    files = {a.path for a in mlflow.artifacts.list_artifacts(run_id=run.info.run_id)}
    assert files == {"threshold.json", "figures"}
    figures = mlflow.artifacts.list_artifacts(run_id=run.info.run_id, artifact_path="figures")
    assert {Path(a.path).name for a in figures} == {p.name for p in out.figures.iterdir()}
    # Registering is not the stage's job: re-running it must never add a registry version.
    assert mlflow.search_logged_models(output_format="list") == []
    assert mlflow.MlflowClient().search_registered_models() == []


def test_export_writes_an_onnx_model_that_scores_like_pytorch(fake_project: Path) -> None:
    train_patchcore.main(["--category", "bottle"])
    evaluate_patchcore.main(["--category", "bottle"])
    export_patchcore.main(["--category", "bottle"])

    path = export_patchcore.onnx_path("bottle")
    threshold = json.loads(Outputs.for_category("bottle").threshold.read_text())["threshold"]
    assert read_metadata(path) == {
        "category": "bottle",
        "image_size": "64",  # from model.pt, the size the model was trained at
        "threshold": str(threshold),
        "git_sha": git_sha(),
    }
    report = json.loads(export_patchcore.report_path("bottle").read_text())
    for name in ("image_auroc", "aupro"):
        assert report[f"onnx_{name}"] == pytest.approx(report[f"pytorch_{name}"], abs=1e-3)
    metrics = json.loads(Outputs.for_category("bottle").metrics.read_text())
    assert report["pytorch_aupro"] == metrics["holdout_aupro"]
    assert report["onnx_mb"] == pytest.approx(path.stat().st_size / 1e6)


def test_export_fails_when_onnx_scores_differently(fake_project: Path) -> None:
    train_patchcore.main(["--category", "bottle"])
    evaluate_patchcore.main(["--category", "bottle"])
    metrics_path = Outputs.for_category("bottle").metrics
    metrics = json.loads(metrics_path.read_text())
    metrics["holdout_aupro"] += 0.002  # as if PyTorch had scored differently
    metrics_path.write_text(json.dumps(metrics))
    with pytest.raises(ValueError, match="scores differently"):
        export_patchcore.main(["--category", "bottle"])


def test_optimize_writes_an_int8_model_and_one_speed_table(
    fake_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for stage in (train_patchcore, evaluate_patchcore, export_patchcore):
        stage.main(["--category", "bottle"])
    calibrated_on: list[pd.DataFrame] = []
    real = optimize_patchcore.calibration_images

    def spy(train: pd.DataFrame, count: int, size: int) -> Any:
        calibrated_on.append(train)
        return real(train, count, size)

    monkeypatch.setattr(optimize_patchcore, "calibration_images", spy)
    optimize_patchcore.main(["--category", "bottle"])

    (rows,) = calibrated_on
    assert set(rows["split"]) == {"train"}  # never calib or holdout images
    assert set(rows["category"]) == {"bottle"}
    assert optimize_patchcore.int8_path("bottle").exists()
    report = json.loads(optimize_patchcore.report_path("bottle").read_text())
    exported = json.loads(export_patchcore.report_path("bottle").read_text())
    assert report["threads"] == 4
    assert report["pytorch_fp32"]["aupro"] == exported["pytorch_aupro"]
    assert report["onnx_fp32"]["image_auroc"] == exported["onnx_image_auroc"]
    # The INT8 accuracy is the INT8 file's own, scored on holdout.
    holdout = manifest().query("split == 'holdout'")
    loader = DataLoader(ManifestDataset(holdout, 64), batch_size=8)
    int8 = OnnxPatchCore(optimize_patchcore.int8_path("bottle"))
    found = evaluate(holdout["label"].to_numpy(), predict(int8, loader), load_masks(holdout, 64))
    assert report["onnx_int8"]["aupro"] == pytest.approx(found["aupro"])
    assert report["onnx_int8"]["image_auroc"] == pytest.approx(found["image_auroc"])
    for name in ("pytorch_fp32", "onnx_fp32", "onnx_int8"):
        row = report[name]
        assert 0 < row["p50_ms"] <= row["p95_ms"]
        assert row["images_per_s_batch8"] > 0
    sizes = {name: report[name]["size_mb"] for name in ("pytorch_fp32", "onnx_fp32", "onnx_int8")}
    assert sizes["onnx_int8"] == pytest.approx(
        optimize_patchcore.int8_path("bottle").stat().st_size / 1e6
    )
    assert sizes["onnx_int8"] < sizes["onnx_fp32"]
