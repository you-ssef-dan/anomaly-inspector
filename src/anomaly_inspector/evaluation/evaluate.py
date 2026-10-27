import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from numpy.typing import ArrayLike, NDArray
from PIL import Image

from anomaly_inspector.evaluation.metrics import aupro, image_auroc, pixel_auroc

# images (B, 3, H, W) -> (anomaly maps (B, 1, H, W) or (B, H, W), image scores (B,))
Predictor = Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]


@dataclass(frozen=True)
class Predictions:
    scores: NDArray[np.float32]  # (N,)
    maps: NDArray[np.float32]  # (N, H, W)
    seconds: float  # time spent in the model, batches included

    @property
    def ms_per_image(self) -> float:
        return 1000 * self.seconds / len(self.scores)


def load_masks(rows: pd.DataFrame, size: int) -> NDArray[np.bool_]:
    """Ground-truth masks at size x size, resized with nearest-neighbour so they stay 0 or 1."""
    masks = []
    for path in rows["mask_path"]:
        if isinstance(path, str):
            with Image.open(path) as mask:
                masks.append(np.asarray(mask.resize((size, size), Image.Resampling.NEAREST)) > 0)
        else:
            masks.append(np.zeros((size, size), dtype=bool))
    return np.stack(masks)


@torch.no_grad()
def predict(model: Predictor, loader: Iterable[tuple[torch.Tensor, torch.Tensor]]) -> Predictions:
    """Run a model over every batch, keeping the order of the rows."""
    maps, scores, seconds = [], [], 0.0
    for images, _ in loader:
        start = time.perf_counter()
        anomaly_map, score = model(images)
        seconds += time.perf_counter() - start
        maps.append(anomaly_map.reshape(-1, *anomaly_map.shape[-2:]).float().numpy())
        scores.append(score.float().numpy())
    return Predictions(np.concatenate(scores), np.concatenate(maps), seconds)


def resize_maps(maps: NDArray[np.float32], size: int) -> NDArray[np.float32]:
    """Bilinear resize of (N, H, W) maps, so a model run at 320 px is scored on 256 px masks."""
    if maps.shape[-2:] == (size, size):
        return maps
    resized = F.interpolate(torch.from_numpy(maps)[:, None], size=(size, size), mode="bilinear")
    return resized[:, 0].numpy()


def evaluate(
    labels: ArrayLike, predictions: Predictions, masks: NDArray[np.bool_]
) -> dict[str, float]:
    """Image AUROC, pixel AUROC and AUPRO, with the maps brought to the masks' size first."""
    maps = resize_maps(predictions.maps, masks.shape[-1])
    return {
        "image_auroc": image_auroc(labels, predictions.scores),
        "pixel_auroc": pixel_auroc(masks, maps),
        "aupro": aupro(masks, maps),
    }
