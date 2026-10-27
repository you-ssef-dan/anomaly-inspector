import numpy as np
from numpy.typing import ArrayLike, NDArray


def lowest_scoring_defects(labels: ArrayLike, scores: ArrayLike, k: int) -> NDArray[np.intp]:
    """Positions of the k defective images with the lowest scores: the closest to being missed.

    Needs no threshold. Whatever threshold is chosen later, a defect misses it only after all
    the defects scoring even lower have. Ties keep their original order.
    """
    labels_a, scores_a = np.asarray(labels), np.asarray(scores)
    defective = np.flatnonzero(labels_a == 1)
    return defective[np.argsort(scores_a[defective], kind="stable")][:k]


def highest_scoring_goods(labels: ArrayLike, scores: ArrayLike, k: int) -> NDArray[np.intp]:
    """Positions of the k good images with the highest scores: the closest to a false reject."""
    labels_a, scores_a = np.asarray(labels), np.asarray(scores)
    good = np.flatnonzero(labels_a == 0)
    return good[np.argsort(-scores_a[good], kind="stable")][:k]


def rank_among(labels: ArrayLike, scores: ArrayLike, position: int, *, highest: bool) -> int:
    """1-based rank of one image among the images of its own class, 1 = the most extreme.

    For a defect `highest=False` ranks from the lowest score; for a good image `highest=True`
    ranks from the highest. It tells how unusual the failure is within its class.
    """
    labels_a, scores_a = np.asarray(labels), np.asarray(scores)
    same = scores_a[labels_a == labels_a[position]]
    return int(
        (same > scores_a[position]).sum() + 1 if highest else (same < scores_a[position]).sum() + 1
    )


def defect_percent(mask: ArrayLike) -> float:
    """Share of the image the defect covers, in percent."""
    return 100 * float(np.asarray(mask, dtype=bool).mean())
