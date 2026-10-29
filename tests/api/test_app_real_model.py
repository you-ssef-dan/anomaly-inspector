"""The real app, the real Inspector and a real (tiny) ONNX model, end to end."""

import io
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from anomaly_inspector.export.onnx_export import export_onnx, write_metadata
from anomaly_inspector.models.backbones import build_backbone
from anomaly_inspector.models.patchcore import PatchCore
from anomaly_inspector.serving.app import create_app
from anomaly_inspector.serving.inference import Inspector
from anomaly_inspector.serving.settings import Settings

SIZE = 64


@pytest.fixture(scope="module")
def model_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    torch.manual_seed(0)
    model = PatchCore(build_backbone("resnet18", pretrained=False))
    images = torch.rand(4, 3, SIZE, SIZE, generator=torch.Generator().manual_seed(1))
    model.fit([(images, torch.zeros(4))], coreset_ratio=0.5, seed=0)
    path = export_onnx(model, tmp_path_factory.mktemp("model") / "model.onnx", SIZE)
    metadata = {"threshold": "1.5", "image_size": str(SIZE), "category": "bottle", "git_sha": "abc"}
    write_metadata(path, metadata)
    return path


@pytest.fixture
def client(model_path: Path) -> Iterator[TestClient]:
    with TestClient(create_app(Settings(model_path=model_path, threads=1))) as test_client:
        yield test_client


def test_the_api_answers_what_the_model_computes(client: TestClient, model_path: Path) -> None:
    pixels = np.random.default_rng(5).integers(0, 256, (80, 80, 3), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG")

    response = client.post(
        "/v1/predict",
        params={"heatmap": True},
        files={"file": ("part.png", buffer.getvalue(), "image/png")},
    )

    assert response.status_code == 200
    body = response.json()
    expected = Inspector(model_path).predict(Image.fromarray(pixels))
    assert body["anomaly_score"] == pytest.approx(expected.score, abs=1e-5)
    assert body["is_anomalous"] == expected.is_anomalous
    assert body["heatmap_png_base64"]
    assert client.get("/v1/model").json() == {
        "category": "bottle",
        "threshold": 1.5,
        "image_size": SIZE,
        "model_version": "abc",
    }
