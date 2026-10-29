"""preprocess, for any image it may receive: always (3, S, S), always finite (hypothesis)."""

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from PIL import Image

from anomaly_inspector.preprocessing import preprocess


@settings(max_examples=60, deadline=None)
@given(
    width=st.integers(1, 300),
    height=st.integers(1, 300),
    mode=st.sampled_from(["L", "RGB", "RGBA", "P"]),
    size=st.sampled_from([32, 64, 256]),
    seed=st.integers(0, 2**32 - 1),
)
def test_any_image_becomes_a_finite_3_by_size_by_size_array(
    width: int, height: int, mode: str, size: int, seed: int
) -> None:
    pixels = np.random.default_rng(seed).integers(0, 256, (height, width, 3), dtype=np.uint8)
    result = preprocess(Image.fromarray(pixels).convert(mode), size)
    assert result.shape == (3, size, size)
    assert result.dtype == np.float32
    assert np.isfinite(result).all()
