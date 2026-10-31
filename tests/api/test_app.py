"""The API with a fake model: status codes, the response, the heatmap. No data, no ONNX."""

import base64
import io
from collections.abc import Iterator, Sequence

import numpy as np
import pytest
from fastapi import HTTPException, UploadFile
from fastapi.testclient import TestClient
from PIL import Image
from starlette.datastructures import Headers

from anomaly_inspector.serving.app import create_app, read_image
from anomaly_inspector.serving.inference import Prediction
from anomaly_inspector.serving.settings import Settings

SIZE = 32


class FakeModel:
    """Scores an image by its mean brightness, so tests can steer the decision."""

    threshold, image_size, category, version = 0.5, SIZE, "bottle", "test-sha"

    def __init__(self) -> None:
        self.calls: list[int] = []  # batch size of every model call

    def predict_batch(self, images: Sequence[Image.Image]) -> list[Prediction]:
        self.calls.append(len(images))
        scores = [np.asarray(image.convert("L"), np.float32).mean() / 255 for image in images]
        return [
            Prediction(float(s), float(s) >= self.threshold, np.full((SIZE, SIZE), s, np.float32))
            for s in scores
        ]


@pytest.fixture
def model() -> FakeModel:
    return FakeModel()


@pytest.fixture
def client(model: FakeModel) -> Iterator[TestClient]:
    settings = Settings(max_upload_mb=0.01, max_batch=3)  # 10 KB: small enough to test 413
    with TestClient(create_app(settings, load=lambda _: model)) as test_client:
        yield test_client


def image_bytes(gray: int, fmt: str = "PNG") -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (SIZE, SIZE), (gray, gray, gray)).save(buffer, format=fmt)
    return buffer.getvalue()


def upload(name: str, content: bytes, mime: str = "image/png") -> tuple[str, bytes, str]:
    return (name, content, mime)


def test_a_bright_part_is_rejected_and_a_dark_one_passes(client: TestClient) -> None:
    bright = client.post("/v1/predict", files={"file": upload("bright.png", image_bytes(230))})
    dark = client.post("/v1/predict", files={"file": upload("dark.png", image_bytes(20))})
    assert bright.status_code == dark.status_code == 200
    assert bright.json()["is_anomalous"] is True
    assert dark.json()["is_anomalous"] is False
    body = bright.json()
    assert body["anomaly_score"] == pytest.approx(230 / 255)
    assert (body["threshold"], body["category"], body["model_version"]) == (
        0.5,
        "bottle",
        "test-sha",
    )
    assert body["filename"] == "bright.png"
    assert body["latency_ms"] >= 0
    assert body["heatmap_png_base64"] is None  # only on request


def test_jpeg_is_accepted_too(client: TestClient) -> None:
    jpeg = upload("part.jpg", image_bytes(200, "JPEG"), "image/jpeg")
    assert client.post("/v1/predict", files={"file": jpeg}).status_code == 200


def test_the_heatmap_is_a_png_of_the_models_input_size(client: TestClient) -> None:
    response = client.post(
        "/v1/predict", params={"heatmap": True}, files={"file": upload("p.png", image_bytes(230))}
    )
    png = base64.b64decode(response.json()["heatmap_png_base64"])
    with Image.open(io.BytesIO(png)) as heatmap:
        assert (heatmap.format, heatmap.size, heatmap.mode) == ("PNG", (SIZE, SIZE), "RGB")


@pytest.mark.parametrize(
    ("file", "status", "message"),
    [
        (upload("notes.txt", b"hello", "text/plain"), 415, "PNG or JPEG"),
        (upload("broken.png", b"\x89PNG not really"), 422, "could not decode"),
        (upload("huge.png", b"\x89PNG" + bytes(20_000)), 413, "larger than"),
    ],
)
def test_bad_uploads_get_a_clear_error(
    client: TestClient, model: FakeModel, file: tuple[str, bytes, str], status: int, message: str
) -> None:
    response = client.post("/v1/predict", files={"file": file})
    assert response.status_code == status
    assert message in response.json()["detail"]
    assert file[0] in response.json()["detail"]  # says which file
    assert model.calls == []  # the model never saw it


def test_a_png_with_a_valid_header_but_cut_off_pixels_is_refused(
    client: TestClient, model: FakeModel
) -> None:
    whole = image_bytes(100)
    cut = upload("cut.png", whole[: len(whole) // 2])
    response = client.post("/v1/predict", files={"file": cut})
    assert response.status_code == 422  # not a 500 from deep inside the model
    assert model.calls == []


def test_an_oversized_upload_is_never_read_whole() -> None:
    class Counting(io.BytesIO):
        largest_read = 0

        def read(self, size: int | None = -1) -> bytes:
            Counting.largest_read = max(Counting.largest_read, size if size and size > 0 else 0)
            return super().read(size)

    upload_file = UploadFile(
        Counting(bytes(5_000_000)),
        filename="big.png",
        headers=Headers({"content-type": "image/png"}),
    )
    with pytest.raises(HTTPException) as error:
        read_image(upload_file, max_mb=1)
    assert error.value.status_code == 413
    assert Counting.largest_read == 1024 * 1024 + 1  # the limit plus one byte, never 5 MB


def test_a_missing_file_is_a_validation_error(client: TestClient) -> None:
    assert client.post("/v1/predict").status_code == 422


def test_a_batch_is_one_model_call_and_keeps_the_order(
    client: TestClient, model: FakeModel
) -> None:
    files = [("files", upload(f"{g}.png", image_bytes(g))) for g in (230, 20, 60)]
    response = client.post("/v1/predict/batch", files=files)
    assert response.status_code == 200
    results = response.json()["results"]
    assert [r["filename"] for r in results] == ["230.png", "20.png", "60.png"]
    assert [r["anomaly_score"] for r in results] == pytest.approx([230 / 255, 20 / 255, 60 / 255])
    assert [r["is_anomalous"] for r in results] == [True, False, False]
    assert model.calls == [3]


def test_a_batch_over_the_limit_is_refused(client: TestClient, model: FakeModel) -> None:
    files = [("files", upload(f"{i}.png", image_bytes(100))) for i in range(4)]
    response = client.post("/v1/predict/batch", files=files)
    assert response.status_code == 413
    assert model.calls == []


def test_one_bad_file_fails_the_whole_batch_and_names_it(
    client: TestClient, model: FakeModel
) -> None:
    files = [
        ("files", upload("good.png", image_bytes(100))),
        ("files", upload("notes.txt", b"hello", "text/plain")),
    ]
    response = client.post("/v1/predict/batch", files=files)
    assert response.status_code == 415
    assert "notes.txt" in response.json()["detail"]
    assert model.calls == []


def test_health_and_model_info(client: TestClient) -> None:
    assert client.get("/health/live").json() == {"status": "live", "build": "dev"}
    assert client.get("/health/ready").json() == {"status": "ready", "model_version": "test-sha"}
    assert client.get("/v1/model").json() == {
        "category": "bottle",
        "threshold": 0.5,
        "image_size": SIZE,
        "model_version": "test-sha",
    }


def test_before_the_model_is_loaded_the_api_is_live_but_not_ready() -> None:
    # Without `with`, TestClient does not run startup, so no model is loaded.
    client = TestClient(create_app(Settings(build="a1b2c3d"), load=lambda _: FakeModel()))
    assert client.get("/health/live").json() == {"status": "live", "build": "a1b2c3d"}
    assert client.get("/health/ready").status_code == 503
    assert (
        client.post("/v1/predict", files={"file": upload("p.png", image_bytes(1))}).status_code
        == 503
    )


def test_the_model_is_loaded_once_at_startup_with_the_settings() -> None:
    loads: list[Settings] = []
    settings = Settings(threads=3)

    def load(given: Settings) -> FakeModel:
        loads.append(given)
        return FakeModel()

    with TestClient(create_app(settings, load=load)) as client:
        for _ in range(3):
            client.post("/v1/predict", files={"file": upload("p.png", image_bytes(1))})
    assert loads == [settings]


def test_a_model_that_cannot_load_stops_the_startup() -> None:
    def load(_: Settings) -> FakeModel:
        raise ValueError("model.onnx lacks metadata")

    with pytest.raises(ValueError, match="lacks metadata"), TestClient(create_app(load=load)):
        pass


def test_settings_come_from_api_environment_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_MODEL_PATH", "models/patchcore/screw/model.onnx")
    monkeypatch.setenv("API_THREADS", "4")
    settings = Settings()
    assert (str(settings.model_path), settings.threads) == ("models/patchcore/screw/model.onnx", 4)


def test_the_demo_page_is_served_at_the_root(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    page = response.text
    assert "v1/predict?heatmap=true" in page  # it calls the API with the heatmap
    assert "v1/model" in page
    assert '<input id="file" type="file"' in page
    assert "/" not in client.get("/openapi.json").json()["paths"]  # not an API endpoint
