"""The HTTP API: is this part defective, and where?

  uv run uvicorn anomaly_inspector.serving.app:app      # then open http://localhost:8000/docs

The model is loaded once, when the app starts: if the file is missing or lacks its metadata, the
app does not start at all. The endpoints that run the model are plain `def`, so FastAPI runs them
in a thread pool and one slow image does not hold up the other requests.
"""

import io
import time
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from typing import Annotated, Protocol

from fastapi import Depends, FastAPI, HTTPException, Request, UploadFile
from PIL import Image, UnidentifiedImageError

from anomaly_inspector.serving.heatmap import heatmap_png_base64
from anomaly_inspector.serving.inference import Inspector, Prediction
from anomaly_inspector.serving.schemas import BatchResponse, ModelInfo, PredictResponse
from anomaly_inspector.serving.settings import Settings

ALLOWED_TYPES = {"image/png", "image/jpeg"}


class Model(Protocol):
    """What the API needs from the model. Inspector is one; tests pass a fake."""

    threshold: float
    image_size: int
    category: str
    version: str

    def predict_batch(self, images: Sequence[Image.Image]) -> list[Prediction]: ...


def read_image(file: UploadFile, max_mb: float) -> Image.Image:
    """The upload as an image, or the HTTP error that explains why not."""
    name = file.filename or "the file"
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(415, f"{name}: upload a PNG or JPEG image")
    limit = int(max_mb * 1024 * 1024)
    data = file.file.read(limit + 1)  # never more than the limit plus one byte
    if len(data) > limit:
        raise HTTPException(413, f"{name}: larger than {max_mb:g} MB")
    try:
        image = Image.open(io.BytesIO(data))
        image.load()  # the header alone can look fine; decode the pixels too
    except (UnidentifiedImageError, OSError) as err:
        raise HTTPException(422, f"{name}: could not decode the image") from err
    return image


def load_inspector(settings: Settings) -> Model:
    return Inspector(settings.model_path, settings.threads)


def create_app(
    settings: Settings | None = None, load: Callable[[Settings], Model] = load_inspector
) -> FastAPI:
    """The app. `load` builds the model at startup; tests replace it."""
    config = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.model = load(config)
        yield

    app = FastAPI(
        title="Anomaly Inspector",
        version="1.0.0",
        description="Upload a photo of a part: get its anomaly score, the decision and a heatmap.",
        lifespan=lifespan,
    )

    def loaded_model(request: Request) -> Model:
        model: Model | None = getattr(request.app.state, "model", None)
        if model is None:
            raise HTTPException(503, "The model is not loaded yet")
        return model

    ModelDep = Annotated[Model, Depends(loaded_model)]

    def score(model: Model, files: Sequence[UploadFile], heatmap: bool) -> list[PredictResponse]:
        images = [read_image(file, config.max_upload_mb) for file in files]
        start = time.perf_counter()
        predictions = model.predict_batch(images)
        latency_ms = 1000 * (time.perf_counter() - start)
        return [
            PredictResponse(
                filename=file.filename,
                category=model.category,
                anomaly_score=prediction.score,
                threshold=model.threshold,
                is_anomalous=prediction.is_anomalous,
                model_version=model.version,
                latency_ms=latency_ms,
                heatmap_png_base64=heatmap_png_base64(
                    image, prediction.anomaly_map, model.threshold
                )
                if heatmap
                else None,
            )
            for file, image, prediction in zip(files, images, predictions, strict=True)
        ]

    @app.get("/health/live")
    def live() -> dict[str, str]:
        """The process is up."""
        return {"status": "live"}

    @app.get("/health/ready")
    def ready(model: ModelDep) -> dict[str, str]:
        """The model is loaded: the API can answer predictions."""
        return {"status": "ready", "model_version": model.version}

    @app.get("/v1/model")
    def model_info(model: ModelDep) -> ModelInfo:
        """Which model answers: category, decision threshold, input size and version."""
        return ModelInfo(
            category=model.category,
            threshold=model.threshold,
            image_size=model.image_size,
            model_version=model.version,
        )

    @app.post("/v1/predict")
    def predict(file: UploadFile, model: ModelDep, heatmap: bool = False) -> PredictResponse:
        """One image: its anomaly score, the decision and, on request, the heatmap."""
        return score(model, [file], heatmap)[0]

    @app.post("/v1/predict/batch")
    def predict_batch(
        files: list[UploadFile], model: ModelDep, heatmap: bool = False
    ) -> BatchResponse:
        """Several images in one model call; the results come back in the same order."""
        if len(files) > config.max_batch:
            raise HTTPException(413, f"at most {config.max_batch} images per request")
        return BatchResponse(results=score(model, files, heatmap))

    return app


app = create_app()  # what uvicorn serves; the model loads when it starts
