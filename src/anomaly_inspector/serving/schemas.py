"""What the API answers, as typed models: they also document /docs."""

from pydantic import BaseModel


class PredictResponse(BaseModel):
    filename: str | None
    category: str
    anomaly_score: float
    threshold: float
    is_anomalous: bool  # anomaly_score >= threshold: reject the part
    model_version: str  # git SHA of the commit that exported the model
    latency_ms: float  # the model only, for the whole request (batch included)
    heatmap_png_base64: str | None = None  # the image with the anomaly map over it


class BatchResponse(BaseModel):
    results: list[PredictResponse]  # in the order the files were sent


class ModelInfo(BaseModel):
    category: str
    threshold: float
    image_size: int
    model_version: str
