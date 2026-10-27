"""plots.py: the ROC and PRO curves logged with every run."""

import numpy as np
import pytest
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from anomaly_inspector.evaluation.metrics import aupro, image_auroc
from anomaly_inspector.evaluation.plots import pro_figure, roc_figure

RNG = np.random.default_rng(0)
LABELS = np.array([0] * 20 + [1] * 30)
SCORES = np.concatenate([RNG.normal(1.8, 0.1, 20), RNG.normal(2.1, 0.2, 30)])
MASKS = np.zeros((6, 32, 32), dtype=bool)
MASKS[:3, 10:20, 10:20] = True
MAPS = RNG.random((6, 32, 32)) + MASKS * 0.6


def the_curve(fig: Figure) -> tuple[np.ndarray, np.ndarray]:
    curve = fig.axes[0].lines[1]  # lines[0] is the chance diagonal
    return np.asarray(curve.get_xdata()), np.asarray(curve.get_ydata())


def test_the_roc_title_gives_the_image_auroc_and_the_curve_runs_corner_to_corner() -> None:
    fig = roc_figure(LABELS, SCORES)
    assert f"{image_auroc(LABELS, SCORES):.3f}" in fig.axes[0].get_title(loc="left")
    x, y = the_curve(fig)
    assert (x[0], y[0], x[-1], y[-1]) == (0, 0, 1, 1)
    assert np.all(np.diff(x) >= 0)


def test_the_pro_curve_starts_at_zero_and_its_area_up_to_the_limit_is_the_aupro() -> None:
    fig = pro_figure(MASKS, MAPS)
    assert f"{aupro(MASKS, MAPS):.3f}" in fig.axes[0].get_title(loc="left")
    x, y = the_curve(fig)
    assert (x[0], y[0]) == (0, 0)
    inside = x <= 0.3
    assert np.trapezoid(y[inside], x[inside]) / 0.3 == pytest.approx(aupro(MASKS, MAPS), abs=0.01)


def test_the_pro_curve_is_drawn_from_a_thousand_points_not_one_per_pixel() -> None:
    x, _ = the_curve(pro_figure(MASKS, MAPS))
    assert len(x) == 1002  # (0, 0) and 1001 evenly spaced points
    assert len(x) < MASKS.size


def test_the_aupro_region_is_shaded_up_to_the_limit() -> None:
    (shade,) = pro_figure(MASKS, MAPS, fpr_limit=0.2).axes[0].patches
    assert isinstance(shade, Rectangle)
    assert shade.get_x() == 0
    assert shade.get_width() == pytest.approx(0.2)
