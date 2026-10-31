"""The API's settings, from environment variables (API_MODEL_PATH, API_THREADS, ...)."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="API_", frozen=True)

    # One API process serves one category's model (each product has its own).
    model_path: Path = Path("models/patchcore/bottle/model.onnx")
    threads: int = Field(2, ge=1)  # ONNX Runtime threads; the container gets 2 CPUs (Phase 7)
    max_upload_mb: float = Field(5, gt=0)  # the largest MVTec image is 2 MB
    max_batch: int = Field(8, ge=1)  # images per /v1/predict/batch request
    # The commit the image was built from (CD sets it), so a deploy can check what is live.
    build: str = "dev"
