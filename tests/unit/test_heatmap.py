"""heatmap.py: the wash follows the threshold, not the image's own hottest spot."""

import numpy as np
import pytest
from PIL import Image

from anomaly_inspector.serving.heatmap import HEAT, heat_alpha, overlay


def test_the_wash_starts_at_90_percent_of_the_threshold_and_is_full_at_it() -> None:
    anomaly_map = np.array([[0.0, 1.8, 1.9, 2.0, 3.0]], np.float32)  # threshold 2.0
    assert heat_alpha(anomaly_map, 2.0)[0].tolist() == pytest.approx([0, 0, 0.4, 0.8, 0.8])


def test_a_part_far_below_the_threshold_shows_no_wash_at_all() -> None:
    # Normalising by the image's own maximum would paint its hottest spot orange anyway.
    anomaly_map = np.linspace(0.1, 1.0, 64, dtype=np.float32).reshape(8, 8)
    image = Image.new("RGB", (16, 16), (40, 90, 140))
    result = np.asarray(overlay(image, anomaly_map, threshold=2.0))
    assert result.shape == (8, 8, 3)  # at the map's size
    assert (result == (40, 90, 140)).all()


def test_above_the_threshold_the_pixel_turns_mostly_orange() -> None:
    image = Image.new("L", (8, 8), 0)  # grayscale in, RGB out
    result = np.asarray(overlay(image, np.full((8, 8), 5.0, np.float32), threshold=2.0))
    np.testing.assert_allclose(result[0, 0], (0.8 * HEAT).round(), atol=1)
