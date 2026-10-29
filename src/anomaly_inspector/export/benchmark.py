"""Latency measured the same way for every model: warm-up first, then percentiles.

The first calls allocate memory and are slower than all the others, so they are not counted.
p95 matters as much as p50 on a production line: one slow image holds up the parts behind it.
"""

import time
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

Run = Callable[[NDArray[np.float32]], object]  # one batch of images in, anything out


def measure(run: Run, batch: int, size: int, runs: int = 30, warmup: int = 5) -> dict[str, float]:
    """p50 and p95 milliseconds per call, and images per second, on a fixed random batch."""
    images = np.random.default_rng(0).random((batch, 3, size, size), dtype=np.float32)
    for _ in range(warmup):
        run(images)
    ms = []
    for _ in range(runs):
        start = time.perf_counter()
        run(images)
        ms.append(1000 * (time.perf_counter() - start))
    return {
        "p50_ms": float(np.percentile(ms, 50)),
        "p95_ms": float(np.percentile(ms, 95)),
        "images_per_s": batch * 1000 / float(np.mean(ms)),
    }
