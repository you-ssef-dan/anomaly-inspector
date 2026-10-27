import numpy as np
import pytest

from anomaly_inspector.evaluation.failures import (
    defect_percent,
    highest_scoring_goods,
    lowest_scoring_defects,
    rank_among,
)

#              position:  0    1    2    3    4    5    6    7
LABELS = np.array([0, 1, 0, 1, 1, 0, 1, 0])
SCORES = np.array([1.0, 3.0, 2.5, 0.5, 2.0, 0.2, 9.0, 1.5])


def test_the_lowest_scoring_defects_come_first() -> None:
    # defects are at 1, 3, 4, 6 with scores 3.0, 0.5, 2.0, 9.0
    assert lowest_scoring_defects(LABELS, SCORES, 2).tolist() == [3, 4]
    assert lowest_scoring_defects(LABELS, SCORES, 10).tolist() == [3, 4, 1, 6]


def test_the_highest_scoring_good_images_come_first() -> None:
    # good images are at 0, 2, 5, 7 with scores 1.0, 2.5, 0.2, 1.5
    assert highest_scoring_goods(LABELS, SCORES, 2).tolist() == [2, 7]
    assert highest_scoring_goods(LABELS, SCORES, 10).tolist() == [2, 7, 0, 5]


def test_selected_positions_point_into_the_original_arrays() -> None:
    for position in lowest_scoring_defects(LABELS, SCORES, 3):
        assert LABELS[position] == 1
    for position in highest_scoring_goods(LABELS, SCORES, 3):
        assert LABELS[position] == 0


def test_ties_keep_their_original_order() -> None:
    labels, scores = [1, 1, 1, 0, 0, 0], [1.0, 1.0, 1.0, 2.0, 2.0, 2.0]
    assert lowest_scoring_defects(labels, scores, 3).tolist() == [0, 1, 2]
    assert highest_scoring_goods(labels, scores, 3).tolist() == [3, 4, 5]


def test_asking_for_zero_or_for_a_class_that_is_absent_gives_nothing() -> None:
    assert lowest_scoring_defects(LABELS, SCORES, 0).tolist() == []
    assert lowest_scoring_defects([0, 0], [1.0, 2.0], 3).tolist() == []
    assert highest_scoring_goods([1, 1], [1.0, 2.0], 3).tolist() == []


def test_accepts_plain_lists() -> None:
    assert lowest_scoring_defects([1, 0, 1], [5.0, 1.0, 2.0], 1).tolist() == [2]


def test_rank_is_counted_inside_the_images_own_class() -> None:
    # the lowest defect (score 0.5) ranks 1st of the defects, although two good images are lower
    assert rank_among(LABELS, SCORES, 3, highest=False) == 1
    assert rank_among(LABELS, SCORES, 6, highest=False) == 4  # the highest defect, from below
    assert rank_among(LABELS, SCORES, 2, highest=True) == 1  # the highest good image
    assert rank_among(LABELS, SCORES, 5, highest=True) == 4


def test_equal_scores_share_a_rank() -> None:
    labels, scores = [0, 0, 0], [2.0, 2.0, 1.0]
    assert rank_among(labels, scores, 0, highest=True) == 1
    assert rank_among(labels, scores, 1, highest=True) == 1
    assert rank_among(labels, scores, 2, highest=True) == 3


@pytest.mark.parametrize(("covered", "expected"), [(0, 0.0), (25, 25.0), (100, 100.0)])
def test_defect_percent(covered: int, expected: float) -> None:
    mask = np.zeros((10, 10), dtype=bool)
    mask.flat[:covered] = True
    assert defect_percent(mask) == pytest.approx(expected)
