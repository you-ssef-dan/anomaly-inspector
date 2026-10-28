"""The DVC stages train_patchcore and evaluate_patchcore, run on the fake dataset."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from anomaly_inspector.config import load_params
from anomaly_inspector.evaluation import thresholds
from anomaly_inspector.models.storage import load_patchcore
from anomaly_inspector.pipelines import evaluate_patchcore, train_patchcore
from anomaly_inspector.pipelines.evaluate_patchcore import Outputs

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
