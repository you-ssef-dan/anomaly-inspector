"""onnx_export.py: the ONNX file computes what PyTorch computes, and carries its metadata."""

from pathlib import Path

import numpy as np
import onnx
import pytest
import torch

from anomaly_inspector.export.onnx_export import (
    OnnxPatchCore,
    export_onnx,
    read_metadata,
    write_metadata,
)
from anomaly_inspector.models.backbones import build_backbone
from anomaly_inspector.models.patchcore import PatchCore

SIZE = 64


@pytest.fixture(scope="module")
def tiny(tmp_path_factory: pytest.TempPathFactory) -> tuple[PatchCore, Path]:
    """A fitted PatchCore on an untrained ResNet-18, and its ONNX export (made once)."""
    torch.manual_seed(0)
    model = PatchCore(build_backbone("resnet18", pretrained=False))
    images = torch.rand(4, 3, SIZE, SIZE, generator=torch.Generator().manual_seed(1))
    model.fit([(images, torch.zeros(4))], coreset_ratio=0.5, seed=0)
    path = export_onnx(model, tmp_path_factory.mktemp("onnx") / "deep" / "model.onnx", SIZE)
    return model.eval(), path


@pytest.mark.parametrize("batch", [1, 3, 8])
def test_onnx_matches_pytorch_at_any_batch_size(tiny: tuple[PatchCore, Path], batch: int) -> None:
    model, path = tiny
    # New images, not the training ones: a patch identical to a stored one sits at distance 0,
    # where the square root magnifies float noise, and no real part is a copy of a training one.
    images = torch.rand(batch, 3, SIZE, SIZE, generator=torch.Generator().manual_seed(100 + batch))
    with torch.no_grad():
        map_pt, score_pt = model(images)
    map_onnx, score_onnx = OnnxPatchCore(path)(images)
    assert map_onnx.shape == (batch, 1, SIZE, SIZE)
    # Float maths runs in a different order in each framework: tight, not bit-exact.
    np.testing.assert_allclose(score_onnx, score_pt, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(map_onnx, map_pt, rtol=1e-3, atol=1e-3)


def test_the_export_is_one_valid_file(tiny: tuple[PatchCore, Path]) -> None:
    _, path = tiny
    assert [p.name for p in path.parent.iterdir()] == ["model.onnx"]  # no external weights file
    onnx.checker.check_model(onnx.load(str(path)))


def test_the_metadata_travels_inside_the_file(tiny: tuple[PatchCore, Path], tmp_path: Path) -> None:
    _, original = tiny
    path = tmp_path / "model.onnx"
    path.write_bytes(original.read_bytes())
    metadata = {"category": "bottle", "image_size": "64", "threshold": "1.75", "git_sha": "abc"}
    write_metadata(path, metadata)
    assert read_metadata(path) == metadata
    images = torch.rand(2, 3, SIZE, SIZE)
    # The model itself is unchanged by the metadata.
    np.testing.assert_array_equal(
        OnnxPatchCore(path)(images)[1], OnnxPatchCore(original)(images)[1]
    )


def test_a_broken_file_is_refused_before_metadata_is_written(tmp_path: Path) -> None:
    path = tmp_path / "model.onnx"
    path.write_bytes(onnx.ModelProto().SerializeToString())  # no graph, no opset
    with pytest.raises(onnx.checker.ValidationError):
        write_metadata(path, {"category": "bottle"})
