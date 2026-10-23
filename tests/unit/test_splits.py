import math

import pandas as pd
import pytest

from anomaly_inspector.data.splits import SPLITS, assign_splits

VAL_FRACTION = 0.1
# category -> (train/good images, {test defect type: image count})
LAYOUT = {
    "bottle": (30, {"good": 8, "broken": 6, "dirty": 4}),
    "screw": (25, {"good": 10, "scratch": 7, "thread": 5}),
}


@pytest.fixture
def manifest() -> pd.DataFrame:
    """A manifest-shaped table; the split logic never opens a file, so no images are needed."""
    rows = []
    for category, (n_train, test_counts) in LAYOUT.items():
        rows += [(category, "train", "good", i) for i in range(n_train)]
        for defect_type, count in test_counts.items():
            rows += [(category, "test", defect_type, i) for i in range(count)]
    df = pd.DataFrame(rows, columns=["category", "source_split", "defect_type", "n"])
    df["label"] = (df["defect_type"] != "good").astype(int)
    df["path"] = (
        df["category"]
        + "/"
        + df["source_split"]
        + "/"
        + df["defect_type"]
        + "/"
        + df["n"].astype(str)
    )
    return df.drop(columns="n")


def test_every_image_gets_exactly_one_known_split(manifest: pd.DataFrame) -> None:
    result = assign_splits(manifest, VAL_FRACTION, seed=42)
    assert len(result) == len(manifest)
    assert result["path"].is_unique
    assert set(result["split"]) == set(SPLITS)
    assert list(result["path"]) == list(manifest["path"])  # same order, nothing lost


def test_input_is_not_modified(manifest: pd.DataFrame) -> None:
    before = manifest.copy()
    assign_splits(manifest, VAL_FRACTION, seed=42)
    pd.testing.assert_frame_equal(manifest, before)


def test_train_images_never_end_up_in_calib_or_holdout_and_vice_versa(
    manifest: pd.DataFrame,
) -> None:
    result = assign_splits(manifest, VAL_FRACTION, seed=42)
    from_train = result["split"].isin(["train", "val_normal"])
    assert (result.loc[from_train, "source_split"] == "train").all()
    assert (result.loc[~from_train, "source_split"] == "test").all()
    assert (result.loc[from_train, "label"] == 0).all()


def test_val_normal_size_follows_the_fraction_in_every_category(manifest: pd.DataFrame) -> None:
    result = assign_splits(manifest, VAL_FRACTION, seed=42)
    for category, (n_train, _) in LAYOUT.items():
        in_category = result[result["category"] == category]
        n_val = (in_category["split"] == "val_normal").sum()
        assert n_val == math.ceil(VAL_FRACTION * n_train)


def test_calib_and_holdout_split_every_defect_type_in_half(manifest: pd.DataFrame) -> None:
    result = assign_splits(manifest, VAL_FRACTION, seed=42)
    test = result[result["source_split"] == "test"]
    counts = test.groupby(["category", "defect_type", "split"]).size().unstack(fill_value=0)
    assert (counts["calib"] >= 1).all()
    assert (counts["holdout"] >= 1).all()
    assert ((counts["calib"] - counts["holdout"]).abs() <= 1).all()


def test_same_seed_gives_the_same_split_and_another_seed_a_different_one(
    manifest: pd.DataFrame,
) -> None:
    first = assign_splits(manifest, VAL_FRACTION, seed=42)
    again = assign_splits(manifest, VAL_FRACTION, seed=42)
    other = assign_splits(manifest, VAL_FRACTION, seed=7)
    pd.testing.assert_frame_equal(first, again)
    assert not first["split"].equals(other["split"])


def test_adds_only_the_split_column(manifest: pd.DataFrame) -> None:
    result = assign_splits(manifest, VAL_FRACTION, seed=42)
    assert list(result.columns) == [*manifest.columns, "split"]
