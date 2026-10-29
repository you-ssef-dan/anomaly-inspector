"""inference.py: the Inspector reads its settings from the model and decides like the pipeline."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from anomaly_inspector.export.onnx_export import export_onnx, write_metadata
from anomaly_inspector.models.backbones import build_backbone
from anomaly_inspector.models.patchcore import PatchCore
from anomaly_inspector.preprocessing import preprocess
from anomaly_inspector.serving.inference import Inspector

SIZE = 64
METADATA = {"threshold": "1.5", "image_size": str(SIZE), "category": "bottle", "git_sha": "abc"}


@pytest.fixture(scope="module")
def tiny(tmp_path_factory: pytest.TempPathFactory) -> tuple[PatchCore, Path]:
    """A fitted tiny PatchCore and its ONNX export, without metadata yet (made once)."""
    torch.manual_seed(0)
    model = PatchCore(build_backbone("resnet18", pretrained=False))
    images = torch.rand(4, 3, SIZE, SIZE, generator=torch.Generator().manual_seed(1))
    model.fit([(images, torch.zeros(4))], coreset_ratio=0.5, seed=0)
    return model.eval(), export_onnx(model, tmp_path_factory.mktemp("onnx") / "model.onnx", SIZE)


def with_metadata(tiny: tuple[PatchCore, Path], folder: Path, **changes: str) -> Path:
    path = folder / "model.onnx"
    path.write_bytes(tiny[1].read_bytes())
    write_metadata(path, METADATA | changes)
    return path


def part(mode: str = "RGB", seed: int = 0) -> Image.Image:
    pixels = np.random.default_rng(seed).integers(0, 256, (80, 80, 3), dtype=np.uint8)
    return Image.fromarray(pixels).convert(mode)


def test_the_settings_come_from_the_model_file(
    tiny: tuple[PatchCore, Path], tmp_path: Path
) -> None:
    inspector = Inspector(with_metadata(tiny, tmp_path))
    assert (inspector.threshold, inspector.image_size) == (1.5, SIZE)
    assert (inspector.category, inspector.version) == ("bottle", "abc")


def test_the_score_and_map_are_the_models(tiny: tuple[PatchCore, Path], tmp_path: Path) -> None:
    model, _ = tiny
    prediction = Inspector(with_metadata(tiny, tmp_path)).predict(part())
    with torch.no_grad():
        anomaly_map, score = model(torch.from_numpy(preprocess(part(), SIZE)[None]))
    assert prediction.score == pytest.approx(float(score[0]), abs=1e-4)
    assert prediction.anomaly_map.shape == (SIZE, SIZE)
    np.testing.assert_allclose(prediction.anomaly_map, anomaly_map[0, 0], atol=1e-3)


def test_a_score_equal_to_the_threshold_is_rejected(
    tiny: tuple[PatchCore, Path], tmp_path: Path
) -> None:
    score = Inspector(with_metadata(tiny, tmp_path)).predict(part()).score
    (tmp_path / "at").mkdir()
    (tmp_path / "above").mkdir()
    at = Inspector(with_metadata(tiny, tmp_path / "at", threshold=repr(score)))
    above = Inspector(with_metadata(tiny, tmp_path / "above", threshold=repr(score + 1e-3)))
    assert at.predict(part()).is_anomalous  # the pipeline's rule: score >= threshold rejects
    assert not above.predict(part()).is_anomalous


@pytest.mark.parametrize("mode", ["L", "RGBA", "P"])
def test_grayscale_transparent_and_palette_images_are_scored_like_rgb(
    tiny: tuple[PatchCore, Path], tmp_path: Path, mode: str
) -> None:
    inspector = Inspector(with_metadata(tiny, tmp_path))
    image = part(mode)
    expected = inspector.predict(image.convert("RGB")).score
    assert inspector.predict(image).score == pytest.approx(expected, abs=1e-5)


def test_a_model_without_metadata_is_refused_at_startup(tiny: tuple[PatchCore, Path]) -> None:
    with pytest.raises(ValueError, match="threshold"):
        Inspector(tiny[1])


def test_serving_never_loads_torch() -> None:
    # The API image has no PyTorch: one stray import would crash it at startup.
    code = (
        "import sys\n"
        "import anomaly_inspector.serving.inference\n"
        "assert 'torch' not in sys.modules, 'serving loaded torch'\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
