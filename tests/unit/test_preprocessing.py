import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from anomaly_inspector.preprocessing import MEAN, STD, preprocess

DATA = Path(__file__).parents[1] / "data"
GOLDEN_INPUT = DATA / "golden_input.png"  # 48x40 synthetic image, never regenerated
GOLDEN_OUTPUT = DATA / "golden_preprocess_32.npy"
GOLDEN_SIZE = 32

# (1 - mean) / std and -mean / std for R, G, B: the well-known ImageNet extremes
WHITE = [2.2489, 2.4286, 2.6400]
BLACK = [-2.1179, -2.0357, -1.8044]


def solid(color: tuple[int, int, int], size: tuple[int, int] = (20, 12)) -> Image.Image:
    return Image.new("RGB", size, color)


def test_shape_dtype_and_memory_layout() -> None:
    out = preprocess(solid((10, 20, 30)), 16)
    assert out.shape == (3, 16, 16)
    assert out.dtype == np.float32
    assert out.flags["C_CONTIGUOUS"]  # a transposed view would break torch's .view()


@pytest.mark.parametrize("size", [(48, 40), (40, 48), (7, 7), (300, 100)])
def test_any_input_shape_gives_a_square_output(size: tuple[int, int]) -> None:
    assert preprocess(solid((1, 2, 3), size), 32).shape == (3, 32, 32)


def test_black_and_white_give_the_hand_computed_values() -> None:
    black = preprocess(solid((0, 0, 0)), 8)
    white = preprocess(solid((255, 255, 255)), 8)
    np.testing.assert_allclose(black[:, 0, 0], BLACK, atol=1e-4)
    np.testing.assert_allclose(white[:, 0, 0], WHITE, atol=1e-4)
    assert (black == black[:, :1, :1]).all()  # a solid image stays solid


def test_channel_order_is_rgb() -> None:
    red = preprocess(solid((255, 0, 0)), 8)
    np.testing.assert_allclose(red[:, 0, 0], [WHITE[0], BLACK[1], BLACK[2]], atol=1e-4)


def test_grayscale_becomes_three_equal_channels() -> None:
    # screw images are grayscale; undo the normalization and the three channels must agree
    out = preprocess(Image.new("L", (16, 16), 128), 8)
    restored = out * STD[:, None, None] + MEAN[:, None, None]
    np.testing.assert_allclose(restored[0], restored[1], atol=1e-6)
    np.testing.assert_allclose(restored[0], restored[2], atol=1e-6)
    np.testing.assert_allclose(restored.mean(), 128 / 255, atol=1e-6)


def test_alpha_channel_is_dropped() -> None:
    transparent_red = Image.new("RGBA", (16, 16), (255, 0, 0, 0))
    np.testing.assert_array_equal(
        preprocess(transparent_red, 8), preprocess(solid((255, 0, 0), (16, 16)), 8)
    )


def test_input_image_is_not_modified() -> None:
    with Image.open(GOLDEN_INPUT) as image:
        before = image.tobytes()
        preprocess(image, 16)
        assert image.tobytes() == before
        assert image.size == (48, 40)


def test_output_is_deterministic() -> None:
    with Image.open(GOLDEN_INPUT) as image:
        np.testing.assert_array_equal(preprocess(image, 32), preprocess(image, 32))


def test_output_matches_the_golden_file() -> None:
    with Image.open(GOLDEN_INPUT) as image:
        actual = preprocess(image, GOLDEN_SIZE)
    if os.environ.get("UPDATE_GOLDEN") == "1":
        np.save(GOLDEN_OUTPUT, actual)
        pytest.skip(f"rewrote {GOLDEN_OUTPUT.name}: review the change, then commit it")
    np.testing.assert_allclose(
        actual,
        np.load(GOLDEN_OUTPUT),
        rtol=0,
        atol=1e-6,
        err_msg=(
            "preprocess() changed. Every model trained so far saw the old output. If the change "
            "is intended: UPDATE_GOLDEN=1 uv run pytest tests/unit/test_preprocessing.py"
        ),
    )


def test_matches_the_torchvision_imagenet_pipeline() -> None:
    transforms = pytest.importorskip("torchvision.transforms")
    pipeline = transforms.Compose(
        [
            transforms.Resize((GOLDEN_SIZE, GOLDEN_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN.tolist(), STD.tolist()),
        ]
    )
    with Image.open(GOLDEN_INPUT) as image:
        expected = pipeline(image.convert("RGB")).numpy()
        actual = preprocess(image, GOLDEN_SIZE)
    np.testing.assert_allclose(actual, expected, atol=1e-6)


def test_importing_it_needs_only_numpy_and_pillow() -> None:
    heavy = ["torch", "torchvision", "pandas", "scipy", "sklearn", "onnxruntime"]
    code = (
        "import sys, anomaly_inspector.preprocessing; "
        f"print(sorted(m for m in {heavy} if m in sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "[]"
