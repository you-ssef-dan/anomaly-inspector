"""gallery.py: the heat wash and the layout of the picture."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from anomaly_inspector.evaluation.gallery import (
    HEAT,
    Panel,
    draw_gallery,
    failure_panels,
    heat_overlay,
)

ONE_MB = 1_000_000  # the pre-commit hook refuses larger files


def test_the_wash_is_invisible_on_normal_areas_and_strongest_on_the_hottest() -> None:
    heat = np.array([[0.0, 0.5, 1.0, 7.0, -3.0]], dtype=np.float32)
    rgba = heat_overlay(heat)
    assert rgba.shape == (1, 5, 4)
    assert rgba[0, :, 3].tolist() == pytest.approx([0.0, 0.4, 0.8, 0.8, 0.0])  # clipped to [0, 1]
    assert np.allclose(rgba[..., :3], HEAT)


def test_a_gallery_is_one_picture_with_a_row_per_failure_kind(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)

    def panel(defect: bool) -> Panel:
        mask = np.zeros((64, 64), dtype=bool)
        mask[10:20, 10:20] = True
        return Panel(
            image=rng.integers(0, 255, (64, 64, 3), dtype=np.uint8),
            heat=rng.random((64, 64)).astype(np.float32),
            mask=mask if defect else None,
            caption="a caption\nscore 1.00",
        )

    path = draw_gallery(
        [[panel(True)] * 3, [panel(False)] * 2], "title", "subtitle", tmp_path / "deep" / "g.jpg"
    )
    assert path.exists()
    assert 5_000 < path.stat().st_size < ONE_MB
    with Image.open(path) as image:
        assert image.width > image.height * 0.8  # laid out as a grid, not a strip


def test_the_colour_scale_runs_from_a_typical_good_part_to_the_strongest_map_shown(
    tmp_path: Path,
) -> None:
    paths = []
    for i in range(2):
        paths.append(str(tmp_path / f"{i}.png"))
        Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(paths[-1])
    rows = pd.DataFrame({"path": paths, "defect_type": ["good", "crack"]})
    maps = np.full((2, 8, 8), 1.0, dtype=np.float32)
    maps[0, 0, 0] = 1.5  # the good part: one warm pixel
    maps[1] = 2.0  # the defect: warm everywhere, hottest at its centre
    maps[1, 4, 4] = 3.0
    masks = np.zeros((2, 8, 8), dtype=bool)
    masks[1, 3:6, 3:6] = True

    (missed, rejected), records = failure_panels(
        rows, [0, 1], maps.reshape(2, -1).max(axis=1), maps, masks, k=1
    )

    assert missed[0].heat.max() == 1  # the strongest map shown
    assert missed[0].heat[0, 0] == pytest.approx(0.5)  # (2 - 1) / (3 - 1)
    assert rejected[0].heat[0, 0] == pytest.approx(0.25)  # (1.5 - 1) / (3 - 1)
    assert rejected[0].heat[1, 1] == 0  # a typical good pixel shows the image untouched
    assert (records[0]["kind"], records[0]["peak_on_defect"]) == ("missed", True)
