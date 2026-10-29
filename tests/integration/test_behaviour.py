"""The real bottle model behaves as a defect detector should. Needs `dvc pull` (model and data).

make test-slow
"""

from pathlib import Path

import pandas as pd
import pytest
from PIL import Image, ImageDraw

from anomaly_inspector.serving.inference import Inspector

REPO = Path(__file__).parents[2]
MODEL = REPO / "models/patchcore/bottle/model.onnx"
MANIFEST = REPO / "data/processed/manifest.parquet"

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not (MODEL.exists() and MANIFEST.exists()), reason="needs dvc pull"),
]


@pytest.fixture(scope="module")
def inspector() -> Inspector:
    return Inspector(MODEL)


def holdout_image(label: int, defect_type: str) -> Image.Image:
    rows = pd.read_parquet(MANIFEST)
    row = rows[
        (rows["category"] == "bottle")
        & (rows["split"] == "holdout")
        & (rows["label"] == label)
        & (rows["defect_type"] == defect_type)
    ].iloc[0]
    with Image.open(REPO / row["path"]) as image:
        return image.convert("RGB")


def test_a_good_bottle_passes(inspector: Inspector) -> None:
    assert not inspector.predict(holdout_image(0, "good")).is_anomalous


def test_a_broken_bottle_is_rejected(inspector: Inspector) -> None:
    assert inspector.predict(holdout_image(1, "broken_large")).is_anomalous


def test_a_dark_patch_on_a_good_bottle_raises_its_score_past_the_threshold(
    inspector: Inspector,
) -> None:
    good = holdout_image(0, "good")
    patched = good.copy()
    ImageDraw.Draw(patched).rectangle((400, 400, 480, 480), fill=(0, 0, 0))  # a 900 px image
    before, after = inspector.predict(good), inspector.predict(patched)
    assert after.score > before.score
    assert after.is_anomalous
    # ...and the map is hottest inside the patch (at 256 px: about 114 to 137).
    row, col = divmod(int(after.anomaly_map.argmax()), after.anomaly_map.shape[1])
    assert 105 <= row <= 145
    assert 105 <= col <= 145
