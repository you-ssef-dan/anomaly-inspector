import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import ndimage
from sklearn.metrics import roc_auc_score


def image_auroc(labels: ArrayLike, scores: ArrayLike) -> float:
    """How often a defective image scores above a good one: 1 is perfect, 0.5 is chance."""
    return float(roc_auc_score(labels, scores))


def pixel_auroc(masks: ArrayLike, maps: ArrayLike) -> float:
    """AUROC over all pixels. Large defects have many pixels, so they dominate this number."""
    return float(roc_auc_score(np.asarray(masks, dtype=bool).ravel(), np.asarray(maps).ravel()))


def pro_curve(masks: ArrayLike, maps: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The per-region overlap (PRO) curve: (false positive rate, PRO) at every threshold.

    masks and maps have shape (N, H, W). Every connected defect region (4-connected, per image)
    gets the same weight, however small it is. The curve starts at (0, 0) and ends at (1, 1).
    """
    masks_b, maps_f = np.asarray(masks, dtype=bool), np.asarray(maps, dtype=np.float64)
    if masks_b.ndim != 3 or masks_b.shape != maps_f.shape:
        raise ValueError(
            f"masks and maps must both be (N, H, W), got {masks_b.shape} and {maps_f.shape}"
        )

    # Number every defect region across all images; 0 means "not a defect pixel".
    regions = np.zeros(masks_b.shape, dtype=np.int64)
    n_regions = 0
    for i, mask in enumerate(masks_b):
        labelled, n = ndimage.label(mask)
        regions[i] = np.where(labelled > 0, labelled + n_regions, 0)
        n_regions += n
    n_normal = int((~masks_b).sum())
    if n_regions == 0 or n_normal == 0:
        raise ValueError("PRO needs at least one defect region and at least one normal pixel")

    # Lowering the threshold past a pixel adds its share to the curve: a normal pixel adds
    # 1 / n_normal to the FPR, a defect pixel adds 1 / (its region's size * n_regions) to PRO.
    region = regions.ravel()
    size = np.bincount(region, minlength=n_regions + 1)
    pro_step = np.where(region > 0, 1.0 / (size[region] * n_regions), 0.0)
    fpr_step = (region == 0) / n_normal

    order = np.argsort(-maps_f.ravel(), kind="stable")
    scores = maps_f.ravel()[order]
    fpr, pro = np.cumsum(fpr_step[order]), np.cumsum(pro_step[order])
    last_of_tie = np.append(scores[1:] != scores[:-1], True)  # equal scores pass together
    fpr = np.clip(np.concatenate([[0.0], fpr[last_of_tie]]), 0, 1)
    pro = np.clip(np.concatenate([[0.0], pro[last_of_tie]]), 0, 1)
    return fpr, pro


def aupro(masks: ArrayLike, maps: ArrayLike, fpr_limit: float = 0.3) -> float:
    """Area under the PRO curve from FPR 0 to fpr_limit, divided by fpr_limit.

    1 is perfect, 0 means no defect is found before fpr_limit. A constant or random map scores
    fpr_limit / 2 (0.15 by default): chance level, as 0.5 is for AUROC.
    """
    if not 0 < fpr_limit <= 1:
        raise ValueError(f"fpr_limit must be in (0, 1], got {fpr_limit}")
    fpr, pro = pro_curve(masks, maps)
    inside = fpr <= fpr_limit
    j = int(np.argmax(~inside)) if not inside.all() else len(fpr) - 1
    x, y = fpr[inside], pro[inside]
    if x[-1] < fpr_limit:  # close the area exactly at the limit, on the segment that crosses it
        i = j - 1
        y_limit = pro[i] + (pro[j] - pro[i]) * (fpr_limit - fpr[i]) / (fpr[j] - fpr[i])
        x, y = np.append(x, fpr_limit), np.append(y, y_limit)
    return float(np.trapezoid(y, x) / fpr_limit)
