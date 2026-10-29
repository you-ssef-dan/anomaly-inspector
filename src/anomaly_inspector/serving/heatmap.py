"""The heatmap as a PNG: the part, with an orange wash where the model sees an anomaly.

numpy and Pillow only. The wash is measured against the decision threshold, not against the
image itself: a good part stays clear instead of always showing a "hottest" spot.
"""

import base64
import io

import numpy as np
from numpy.typing import NDArray
from PIL import Image

HEAT = np.array([0xEB, 0x68, 0x34], dtype=np.float32)  # the orange of the project's charts
# No wash below 90% of the threshold, full wash from the threshold up. Measured on holdout: from
# 75%, the wash covered 78 to 94% of good screws and carpets; from 90%, 4 to 12% of good parts,
# and still 94 to 100% of each defect; from 100%, only 58% of screw defects.
START = 0.9


def heat_alpha(anomaly_map: NDArray[np.float32], threshold: float) -> NDArray[np.float32]:
    """Opacity of the wash per pixel: 0 below START x threshold, 0.8 from the threshold up."""
    level = (anomaly_map - START * threshold) / ((1 - START) * threshold)
    alpha: NDArray[np.float32] = (0.8 * np.clip(level, 0, 1)).astype(np.float32)
    return alpha


def overlay(image: Image.Image, anomaly_map: NDArray[np.float32], threshold: float) -> Image.Image:
    """The image at the map's size, with the wash blended over it."""
    size = anomaly_map.shape[::-1]  # (width, height)
    rgb = np.asarray(image.convert("RGB").resize(size, Image.Resampling.BILINEAR), np.float32)
    alpha = heat_alpha(anomaly_map, threshold)[..., None]
    return Image.fromarray(((1 - alpha) * rgb + alpha * HEAT).round().astype(np.uint8))


def heatmap_png_base64(
    image: Image.Image, anomaly_map: NDArray[np.float32], threshold: float
) -> str:
    buffer = io.BytesIO()
    overlay(image, anomaly_map, threshold).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")
