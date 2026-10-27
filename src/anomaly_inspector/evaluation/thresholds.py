"""The decision threshold: a part whose score is at or above it is rejected.

AUROC and AUPRO judge the scores without deciding anything. A production line has to decide,
part by part, so the model ships with one threshold, chosen on calib or val_normal (never on
holdout, whose numbers must stay honest).
"""

import numpy as np
from numpy.typing import ArrayLike


def cost_threshold(
    labels: ArrayLike, scores: ArrayLike, miss_cost: float, false_reject_cost: float
) -> float:
    """The threshold with the lowest total cost on parts whose labels are known (calib).

    Every observed score is tried as the threshold. Cost = miss_cost x missed defects +
    false_reject_cost x rejected good parts. On a tie the lowest threshold wins: it catches the
    most defects for the same cost.
    """
    defective = np.asarray(labels) == 1
    scores_a = np.asarray(scores, dtype=np.float64)
    if not defective.any() or defective.all():
        raise ValueError("a cost threshold needs both good and defective parts")
    candidates = np.unique(scores_a)  # sorted, lowest first
    costs = [
        miss_cost * int((defective & (scores_a < t)).sum())
        + false_reject_cost * int((~defective & (scores_a >= t)).sum())
        for t in candidates
    ]
    return float(candidates[int(np.argmin(costs))])


def normal_threshold(good_scores: ArrayLike, quantile: float) -> float:
    """The score that a share `quantile` of good parts stays below (val_normal, no defects).

    For a new product line with no defect images yet. With 0.99 it aims for about 1% false
    rejects, but it says nothing about how many defects it misses.
    """
    scores_a = np.asarray(good_scores, dtype=np.float64)
    if scores_a.size == 0:
        raise ValueError("a normal threshold needs at least one good score")
    return float(np.quantile(scores_a, quantile))
