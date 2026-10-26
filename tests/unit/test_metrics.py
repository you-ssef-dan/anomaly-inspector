import numpy as np
import pytest
from numpy.typing import NDArray
from scipy import ndimage

from anomaly_inspector.evaluation.metrics import aupro, image_auroc, pixel_auroc, pro_curve


def three_defects() -> NDArray[np.bool_]:
    """Four 32 x 32 masks: three defect regions of different sizes, and one good image."""
    masks = np.zeros((4, 32, 32), dtype=bool)
    masks[0, 2:6, 2:6] = True
    masks[1, 10:20, 5:9] = True
    masks[3, 20:30, 20:30] = True
    return masks


def brute_force_aupro(masks: NDArray[np.bool_], maps: NDArray[np.float64], limit: float) -> float:
    """The definition, one threshold at a time: slow, but nothing clever to get wrong."""
    regions = []
    for i, mask in enumerate(masks):
        labelled, n = ndimage.label(mask)
        regions += [(i, labelled == r) for r in range(1, n + 1)]
    normal = ~masks
    points = [(0.0, 0.0)]
    for t in np.unique(maps)[::-1]:
        predicted = maps >= t
        fpr = (predicted & normal).sum() / normal.sum()
        points.append((fpr, np.mean([predicted[i][r].mean() for i, r in regions])))
    fprs, pros = map(np.array, zip(*points, strict=True))
    inside = fprs <= limit
    x = np.append(fprs[inside], limit)
    y = np.append(pros[inside], np.interp(limit, fprs, pros))
    return float(np.trapezoid(y, x) / limit)


# --- image and pixel AUROC -----------------------------------------------------------------


def test_image_auroc_hand_computed_cases() -> None:
    labels = [0, 0, 1, 1]
    assert image_auroc(labels, [0.1, 0.2, 0.8, 0.9]) == 1.0  # every defect above every good
    assert image_auroc(labels, [0.9, 0.8, 0.2, 0.1]) == 0.0  # reversed
    assert image_auroc(labels, [0.5, 0.5, 0.5, 0.5]) == 0.5  # no information: chance
    # 4 (defect, good) pairs; only 0.35 vs 0.4 is in the wrong order: 3 / 4
    assert image_auroc(labels, [0.1, 0.4, 0.35, 0.8]) == 0.75


def test_pixel_auroc_perfect_and_inverted_maps() -> None:
    masks = three_defects()
    assert pixel_auroc(masks, masks.astype(float)) == 1.0
    assert pixel_auroc(masks, 1.0 - masks) == 0.0


# --- AUPRO -----------------------------------------------------------------------------------


def test_aupro_of_a_perfect_map_is_one() -> None:
    masks = three_defects()
    assert aupro(masks, masks.astype(float)) == pytest.approx(1.0)


def test_aupro_of_an_inverted_map_is_zero() -> None:
    masks = three_defects()
    assert aupro(masks, 1.0 - masks) == pytest.approx(0.0)


def test_aupro_of_a_constant_map_is_chance_level() -> None:
    # A constant map cannot rank anything. Like AUROC's 0.5, AUPRO's chance level is
    # fpr_limit / 2, because the curve then runs straight from (0, 0) to (1, 1).
    masks = three_defects()
    assert aupro(masks, np.zeros(masks.shape)) == pytest.approx(0.15)
    assert aupro(masks, np.zeros(masks.shape), fpr_limit=1.0) == pytest.approx(0.5)


def test_aupro_of_random_maps_is_close_to_chance_level() -> None:
    masks = np.zeros((50, 64, 64), dtype=bool)
    masks[:, 10:30, 10:30] = True
    maps = np.random.default_rng(0).random(masks.shape)
    assert aupro(masks, maps) == pytest.approx(0.15, abs=0.01)


def test_every_defect_region_counts_the_same_however_small() -> None:
    # A 4-pixel defect is found perfectly and a 400-pixel one is missed completely.
    masks = np.zeros((1, 40, 40), dtype=bool)
    masks[0, 0:2, 0:2] = True
    masks[0, 20:40, 20:40] = True
    maps = np.full(masks.shape, 0.5)  # normal pixels sit in between
    maps[0, 0:2, 0:2] = 1.0
    maps[0, 20:40, 20:40] = 0.0
    assert aupro(masks, maps) == pytest.approx(0.5)  # one of two defects found
    assert pixel_auroc(masks, maps) < 0.02  # pixel AUROC only sees the 400 missed pixels


def test_regions_in_different_images_are_different_regions() -> None:
    # The same spot in two images is two defects: finding one of them is half the job.
    masks = np.zeros((2, 16, 16), dtype=bool)
    masks[:, 4:8, 4:8] = True
    maps = np.full(masks.shape, 0.5)
    maps[0, 4:8, 4:8] = 1.0
    maps[1, 4:8, 4:8] = 0.0
    assert aupro(masks, maps) == pytest.approx(0.5)


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("limit", [0.05, 0.3, 1.0])
def test_aupro_matches_the_brute_force_definition(seed: int, limit: float) -> None:
    rng = np.random.default_rng(seed)
    maps = rng.random((3, 12, 12)).round(2)  # rounding creates ties on purpose
    masks = np.zeros((3, 12, 12), dtype=bool)
    masks[0, 1:4, 1:4] = True
    masks[1, 5:11, 2:4] = True
    masks[2, 0, 0] = True
    assert aupro(masks, maps, limit) == pytest.approx(brute_force_aupro(masks, maps, limit))


def test_pro_curve_starts_at_zero_ends_at_one_and_only_goes_up() -> None:
    masks = three_defects()
    fpr, pro = pro_curve(masks, np.random.default_rng(0).random(masks.shape))
    assert (fpr[0], pro[0]) == (0.0, 0.0)
    assert fpr[-1] == pytest.approx(1.0)
    assert pro[-1] == pytest.approx(1.0)
    assert (np.diff(fpr) >= 0).all()
    assert (np.diff(pro) >= 0).all()


def test_tied_scores_enter_the_curve_together() -> None:
    masks = three_defects()
    _, pro = pro_curve(masks, np.zeros(masks.shape))
    assert len(pro) == 2  # (0, 0), then every pixel at once: (1, 1)


@pytest.mark.parametrize(
    ("masks", "maps", "message"),
    [
        (np.zeros((2, 8, 8), bool), np.zeros((2, 8, 8)), "defect region"),
        (np.ones((2, 8, 8), bool), np.zeros((2, 8, 8)), "normal pixel"),
        (np.zeros((2, 8, 8), bool), np.zeros((2, 8, 9)), r"\(N, H, W\)"),
        (np.zeros((8, 8), bool), np.zeros((8, 8)), r"\(N, H, W\)"),
    ],
)
def test_impossible_inputs_are_rejected(
    masks: NDArray[np.bool_], maps: NDArray[np.float64], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        aupro(masks, maps)


@pytest.mark.parametrize("limit", [0.0, -0.1, 1.5])
def test_fpr_limit_must_be_a_rate(limit: float) -> None:
    masks = three_defects()
    with pytest.raises(ValueError, match="fpr_limit"):
        aupro(masks, masks.astype(float), fpr_limit=limit)
