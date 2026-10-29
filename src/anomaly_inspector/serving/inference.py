"""The model behind the API: one image in, a score, a decision and a heatmap out.

Only numpy, Pillow and ONNX Runtime: the API image carries no PyTorch. Everything the decision
needs (threshold, image size, category, version) is read from the ONNX file's metadata, written
by the export stage, so nothing here is hard-coded.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray
from PIL import Image

from anomaly_inspector.preprocessing import preprocess

REQUIRED_METADATA = ("threshold", "image_size", "category", "git_sha")


@dataclass(frozen=True)
class Prediction:
    score: float
    is_anomalous: bool  # score >= threshold: the part is rejected
    anomaly_map: NDArray[np.float32]  # (image_size, image_size), higher = more anomalous


class Inspector:
    """Loads the model once; then predict() answers for one image at a time."""

    def __init__(self, model_path: Path, threads: int | None = None) -> None:
        options = ort.SessionOptions()
        if threads is not None:
            options.intra_op_num_threads = threads
        self.session = ort.InferenceSession(
            str(model_path), options, providers=["CPUExecutionProvider"]
        )
        metadata = self.session.get_modelmeta().custom_metadata_map
        missing = [key for key in REQUIRED_METADATA if key not in metadata]
        if missing:  # fail at startup, not at the first request
            raise ValueError(f"{model_path} lacks metadata {missing}: export it with the pipeline")
        self.threshold = float(metadata["threshold"])
        self.image_size = int(metadata["image_size"])
        self.category = metadata["category"]
        self.version = metadata["git_sha"]

    def predict(self, image: Image.Image) -> Prediction:
        # The same preprocessing as in training (a golden-file test pins it), batch of one.
        anomaly_map, score = self.session.run(
            None, {"image": preprocess(image, self.image_size)[None]}
        )
        value = float(score[0])
        return Prediction(
            score=value,
            is_anomalous=value >= self.threshold,
            anomaly_map=anomaly_map[0, 0].astype(np.float32),
        )
