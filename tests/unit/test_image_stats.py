"""Brightness, contrast and sharpness, on images where the answer is known."""

import numpy as np
import pytest
from PIL import Image, ImageFilter

from anomaly_inspector.image_stats import image_stats


def photo(seed: int = 0) -> Image.Image:
    """A color photo: smooth shapes plus noise (pure noise would have no edges to blur), with
    red, green and blue all different."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:96, 0:96]
    shapes = 120 + 60 * np.sin(x / 7) * np.cos(y / 11)
    channels = [shapes, 0.8 * shapes, 0.5 * shapes + 40]
    pixels = np.stack([c + rng.normal(0, 15, shapes.shape) for c in channels], axis=-1)
    return Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))


def test_a_plain_gray_image_has_its_level_as_brightness_and_nothing_else() -> None:
    plain = Image.fromarray(np.full((20, 20), 100, dtype=np.uint8))
    assert image_stats(plain) == {"brightness": 100.0, "contrast": 0.0, "sharpness": 0.0}


def test_the_sharpness_of_a_checkerboard_is_the_variance_of_its_laplacian() -> None:
    # Every pixel's four neighbours have the other value, so the Laplacian is +-4 x 255.
    board = Image.fromarray(((np.indices((8, 8)).sum(axis=0) % 2) * 255).astype(np.uint8))
    stats = image_stats(board)
    assert (stats["brightness"], stats["contrast"]) == (127.5, 127.5)
    assert stats["sharpness"] == (4 * 255) ** 2


def test_a_photo_30_percent_darker_has_30_percent_less_brightness_and_contrast() -> None:
    # What phase 10's lighting test does to the line's photos.
    original = photo()
    darker = Image.eval(original, lambda value: round(value * 0.7))
    before, after = image_stats(original), image_stats(darker)
    assert after["brightness"] == pytest.approx(0.7 * before["brightness"], rel=0.01)
    assert after["contrast"] == pytest.approx(0.7 * before["contrast"], rel=0.01)
    assert after["sharpness"] == pytest.approx(0.49 * before["sharpness"], rel=0.02)


def test_a_blurred_photo_loses_sharpness_but_keeps_its_brightness() -> None:
    original = photo()
    blurred = original.filter(ImageFilter.GaussianBlur(2))
    before, after = image_stats(original), image_stats(blurred)
    assert after["sharpness"] < 0.1 * before["sharpness"]
    assert after["brightness"] == pytest.approx(before["brightness"], rel=0.01)


def test_a_color_photo_is_measured_in_gray_levels() -> None:
    color = photo()
    assert image_stats(color) == image_stats(color.convert("L"))
