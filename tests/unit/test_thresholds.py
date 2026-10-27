"""thresholds.py: the two ways to pick the decision threshold."""

import numpy as np
import pytest

from anomaly_inspector.evaluation.thresholds import cost_threshold, normal_threshold

#              good parts         | defective parts
LABELS = [0, 0, 0, 0, 1, 1, 1]
SCORES = [1.6, 1.7, 1.8, 2.1, 1.75, 2.3, 2.5]


def test_a_cheap_miss_moves_the_threshold_up_and_a_costly_one_down() -> None:
    # Misses as cheap as false rejects: 2.3 misses one defect (1.75) and rejects no good part.
    assert cost_threshold(LABELS, SCORES, miss_cost=1, false_reject_cost=1) == 2.3
    # A miss costs 10: 1.75 catches every defect, for two false rejects (1.8 and 2.1).
    assert cost_threshold(LABELS, SCORES, miss_cost=10, false_reject_cost=1) == 1.75


def test_a_part_scoring_exactly_the_threshold_is_rejected() -> None:
    # Separable scores: the lowest defect (2.0) is the threshold, and score >= 2.0 rejects it.
    assert cost_threshold([0, 0, 1, 1], [1.0, 1.5, 2.0, 3.0], 10, 1) == 2.0


def test_on_a_tie_the_lowest_threshold_wins() -> None:
    # 1.5 (one false reject) and 2.0 (one miss) both cost 1; the lower one catches more defects.
    assert cost_threshold([0, 1, 0, 1], [1.0, 1.5, 1.6, 2.0], 1, 1) == 1.5


@pytest.mark.parametrize("labels", [[0, 0], [1, 1]])
def test_a_cost_threshold_needs_both_kinds_of_part(labels: list[int]) -> None:
    with pytest.raises(ValueError, match="both"):
        cost_threshold(labels, [1.0, 2.0], 10, 1)


def test_the_normal_threshold_leaves_the_chosen_share_of_good_parts_below_it() -> None:
    good = np.arange(1, 101, dtype=float)  # 1, 2, ..., 100
    threshold = normal_threshold(good, 0.9)
    assert threshold == pytest.approx(90.1)  # numpy's linear quantile
    assert (good < threshold).mean() == 0.9


def test_a_normal_threshold_needs_scores() -> None:
    with pytest.raises(ValueError, match="at least one"):
        normal_threshold([], 0.99)
