"""quantize.py and benchmark.py: what goes to 8 bits, and how speed is measured."""

from pathlib import Path

import numpy as np
import onnx
import pytest
import torch

from anomaly_inspector.export.benchmark import measure
from anomaly_inspector.export.onnx_export import OnnxPatchCore, export_onnx
from anomaly_inspector.export.quantize import Images, quantize_int8
from anomaly_inspector.models.backbones import build_backbone
from anomaly_inspector.models.patchcore import PatchCore

SIZE = 64


@pytest.fixture(scope="module")
def models(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """A tiny FP32 export and its INT8 version (made once)."""
    torch.manual_seed(0)
    model = PatchCore(build_backbone("resnet18", pretrained=False))
    images = torch.rand(4, 3, SIZE, SIZE, generator=torch.Generator().manual_seed(1))
    model.fit([(images, torch.zeros(4))], coreset_ratio=0.5, seed=0)
    folder = tmp_path_factory.mktemp("models")
    fp32 = export_onnx(model, folder / "model.onnx", SIZE)
    calibration = [image.numpy() for image in images]
    return fp32, quantize_int8(fp32, folder / "model.int8.onnx", calibration)


def test_the_convolutions_go_to_8_bits_and_the_distances_stay_float(
    models: tuple[Path, Path],
) -> None:
    _, int8 = models
    graph = onnx.load(str(int8)).graph
    made_by = {output: node.op_type for node in graph.node for output in node.output}
    convs = [node for node in graph.node if node.op_type == "Conv"]
    matmuls = [node for node in graph.node if node.op_type == "MatMul"]
    assert convs
    assert matmuls
    # QDQ format: an 8-bit Conv reads its input and weights through DequantizeLinear.
    assert all(made_by.get(node.input[0]) == "DequantizeLinear" for node in convs)
    # The distances to the memory bank: no input comes out of an 8-bit value.
    assert all(made_by.get(name) != "DequantizeLinear" for node in matmuls for name in node.input)


def test_the_int8_file_is_smaller_and_still_runs_at_any_batch_size(
    models: tuple[Path, Path],
) -> None:
    fp32, int8 = models
    assert int8.stat().st_size < 0.5 * fp32.stat().st_size
    for batch in (1, 3):
        anomaly_map, score = OnnxPatchCore(int8)(torch.rand(batch, 3, SIZE, SIZE))
        assert anomaly_map.shape == (batch, 1, SIZE, SIZE)
        assert score.shape == (batch,)
        assert torch.isfinite(score).all()


def test_calibration_images_are_fed_one_at_a_time_then_none() -> None:
    reader = Images(np.zeros((2, 3, SIZE, SIZE), dtype=np.float32))
    first = reader.get_next()
    assert first is not None
    assert first["image"].shape == (1, 3, SIZE, SIZE)
    assert reader.get_next() is not None
    assert reader.get_next() is None


def test_warm_up_calls_are_made_but_not_timed() -> None:
    batches: list[tuple[int, ...]] = []

    def run(images: np.ndarray) -> None:
        batches.append(images.shape)

    result = measure(run, batch=8, size=SIZE, runs=7, warmup=3)
    assert len(batches) == 10
    assert set(batches) == {(8, 3, SIZE, SIZE)}
    assert 0 <= result["p50_ms"] <= result["p95_ms"]
    assert result["images_per_s"] > 0


def test_the_numbers_come_from_the_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    # A clock that moves 5 ms between two readings: every call lasts exactly 5 ms.
    ticks = iter(np.arange(0, 100, 0.005))
    readings: list[float] = []

    def clock() -> float:
        readings.append(next(ticks))
        return readings[-1]

    monkeypatch.setattr("anomaly_inspector.export.benchmark.time.perf_counter", clock)
    result = measure(lambda images: None, batch=8, size=8, runs=10, warmup=2)
    assert len(readings) == 2 * 10  # a start and an end per timed call; warm-up is not timed
    assert result["p50_ms"] == pytest.approx(5)
    assert result["p95_ms"] == pytest.approx(5)
    assert result["images_per_s"] == pytest.approx(8 * 1000 / 5)  # every image of the batch
