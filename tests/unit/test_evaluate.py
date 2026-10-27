from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from anomaly_inspector.data.dataset import ManifestDataset
from anomaly_inspector.data.manifest import build_manifest
from anomaly_inspector.evaluation.evaluate import (
    Predictions,
    evaluate,
    load_masks,
    predict,
    resize_maps,
)
from anomaly_inspector.models.backbones import build_backbone
from anomaly_inspector.models.patchcore import PatchCore


@pytest.fixture
def manifest(fake_mvtec: Path) -> pd.DataFrame:
    return build_manifest(fake_mvtec, ["bottle"])


def test_masks_are_resized_with_nearest_neighbour(manifest: pd.DataFrame) -> None:
    masks = load_masks(manifest, 32)
    assert masks.shape == (len(manifest), 32, 32)
    assert masks.dtype == bool
    # the fake defect is a 10 x 10 square at 64 px: exactly 5 x 5 at 32 px, no blurred edge
    assert [int(m.sum()) for m in masks] == [25 if label else 0 for label in manifest["label"]]


def test_predict_keeps_row_order_across_batches() -> None:
    images = torch.arange(5, dtype=torch.float32)[:, None, None, None].expand(5, 3, 4, 4)

    def model(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return x[:, :1], x[:, 0, 0, 0]  # a map and a score equal to the image's value

    result = predict(model, DataLoader(list(zip(images, range(5), strict=True)), batch_size=2))
    assert result.scores.tolist() == [0, 1, 2, 3, 4]
    assert result.maps.shape == (5, 4, 4)
    assert result.ms_per_image >= 0


def test_resize_maps_keeps_maps_that_already_fit() -> None:
    maps = np.random.default_rng(0).random((2, 8, 8)).astype(np.float32)
    assert resize_maps(maps, 8) is maps
    assert resize_maps(maps, 16).shape == (2, 16, 16)
    constant = np.full((1, 8, 8), 3.0, dtype=np.float32)
    np.testing.assert_allclose(resize_maps(constant, 20), 3.0)


def test_evaluate_scores_maps_of_another_size_on_the_masks_size() -> None:
    masks = np.zeros((4, 16, 16), dtype=bool)
    masks[2:, 4:12, 4:12] = True
    perfect_at_32 = resize_maps(masks.astype(np.float32), 32)  # as if the model ran at 32 px
    predictions = Predictions(np.array([0, 0, 1, 1], dtype=np.float32), perfect_at_32, 1.0)
    metrics = evaluate([0, 0, 1, 1], predictions, masks)
    assert metrics == pytest.approx({"image_auroc": 1.0, "pixel_auroc": 1.0, "aupro": 1.0})


def test_patchcore_twice_with_the_same_seed_gives_identical_metrics(
    manifest: pd.DataFrame,
) -> None:
    # Phase 2 "Done when": a rerun with the same seed reproduces every metric exactly.
    train = DataLoader(ManifestDataset(manifest[manifest["source_split"] == "train"], 64))
    test_rows = manifest[manifest["source_split"] == "test"]
    test = DataLoader(ManifestDataset(test_rows, 64), batch_size=2)
    masks = load_masks(test_rows, 64)

    def run(seed: int) -> dict[str, float]:
        torch.manual_seed(0)
        model = PatchCore(build_backbone("resnet18", pretrained=False))
        model.fit(train, coreset_ratio=0.3, seed=seed)
        return evaluate(test_rows["label"], predict(model, test), masks)

    assert run(seed=1) == run(seed=1)
