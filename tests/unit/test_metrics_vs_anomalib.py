"""Our AUPRO against the reference implementation in anomalib (the optional `bench` extra).

Run with `make test-slow`. Skipped when anomalib is not installed, which is how CI runs.
"""

from importlib.util import find_spec
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter

from anomaly_inspector.evaluation.metrics import aupro

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(find_spec("anomalib") is None, reason="needs the bench extra"),
]


def anomalib_aupro(masks: NDArray[np.bool_], maps: NDArray[np.float32], fpr_limit: float) -> float:
    from anomalib.metrics import AUPRO

    metric = AUPRO(fields=["anomaly_map", "gt_mask"], fpr_limit=fpr_limit)
    metric.update(
        SimpleNamespace(
            anomaly_map=torch.from_numpy(maps).float(), gt_mask=torch.from_numpy(masks).int()
        )
    )
    return float(metric.compute())


def one_square_per_image() -> NDArray[np.bool_]:
    masks = np.zeros((20, 64, 64), dtype=bool)
    masks[:, 20:40, 20:40] = True
    return masks


def three_regions_of_different_size() -> NDArray[np.bool_]:
    masks = np.zeros((10, 96, 96), dtype=bool)
    masks[:, 5:8, 5:8] = True  # 9 pixels
    masks[:, 30:60, 30:70] = True  # 1200 pixels
    masks[:, 80:82, 10:30] = True  # 40 pixels
    return masks


def noisy_signal(masks: NDArray[np.bool_], seed: int) -> NDArray[np.float32]:
    rng = np.random.default_rng(seed)
    return (rng.random(masks.shape) * 0.7 + masks * rng.random(masks.shape) * 0.5).astype(
        np.float32
    )


def smoothed(maps: NDArray[np.float32]) -> NDArray[np.float32]:
    """Blurred like a real anomaly map, which is what the metric will see."""
    return np.stack([gaussian_filter(m, 3) for m in maps]).astype(np.float32)


@pytest.mark.parametrize("fpr_limit", [0.05, 0.3, 1.0])
@pytest.mark.parametrize("kind", ["random", "signal_and_noise", "different_sizes", "smoothed"])
def test_aupro_agrees_with_anomalib(kind: str, fpr_limit: float) -> None:
    rng = np.random.default_rng(0)
    square, regions = one_square_per_image(), three_regions_of_different_size()
    masks, maps = {
        "random": (square, rng.random(square.shape).astype(np.float32)),
        "signal_and_noise": (square, noisy_signal(square, seed=1)),
        "different_sizes": (regions, noisy_signal(regions, seed=2)),
        "smoothed": (regions, smoothed(noisy_signal(regions, seed=2))),
    }[kind]
    assert aupro(masks, maps, fpr_limit) == pytest.approx(
        anomalib_aupro(masks, maps, fpr_limit), abs=1e-5
    )


@pytest.mark.parametrize(
    ("name", "expected"), [("perfect", 1.0), ("inverted", 0.0), ("constant", 0.15)]
)
def test_both_give_the_same_answer_on_the_hand_worked_cases(name: str, expected: float) -> None:
    masks = three_regions_of_different_size()
    maps = {
        "perfect": masks.astype(np.float32),
        "inverted": 1 - masks.astype(np.float32),
        "constant": np.zeros(masks.shape, dtype=np.float32),
    }[name]
    assert aupro(masks, maps) == pytest.approx(expected, abs=1e-6)
    assert anomalib_aupro(masks, maps, 0.3) == pytest.approx(expected, abs=1e-6)
