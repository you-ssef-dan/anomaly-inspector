import hashlib
from pathlib import Path

import pytest

from anomaly_inspector.data.manifest import COLUMNS, build_manifest

CATEGORIES = ["bottle", "screw"]


def test_one_row_per_image_and_masks_are_not_rows(fake_mvtec: Path) -> None:
    df = build_manifest(fake_mvtec, CATEGORIES)
    # bottle 4 + 2 + 2, screw 3 + 1 + 1; the 3 mask files must not appear as rows
    assert len(df) == 13
    assert not df["path"].str.contains("ground_truth").any()


def test_columns_and_order_are_stable(fake_mvtec: Path) -> None:
    assert list(build_manifest(fake_mvtec, CATEGORIES).columns) == COLUMNS
    assert list(build_manifest(fake_mvtec, []).columns) == COLUMNS


def test_split_defect_type_and_label_come_from_the_folders(fake_mvtec: Path) -> None:
    df = build_manifest(fake_mvtec, CATEGORIES)
    counts = df.groupby(["category", "source_split", "defect_type", "label"]).size()
    assert counts.to_dict() == {
        ("bottle", "test", "broken_large", 1): 2,
        ("bottle", "test", "good", 0): 2,
        ("bottle", "train", "good", 0): 4,
        ("screw", "test", "good", 0): 1,
        ("screw", "test", "scratch_head", 1): 1,
        ("screw", "train", "good", 0): 3,
    }


def test_only_defective_images_have_a_mask_and_it_exists(fake_mvtec: Path) -> None:
    df = build_manifest(fake_mvtec, CATEGORIES)
    assert (df["mask_path"].notna() == (df["label"] == 1)).all()
    assert all(Path(p).exists() for p in df["mask_path"].dropna())


def test_paths_are_relative_and_resolve_from_the_repo_root(fake_mvtec: Path) -> None:
    df = build_manifest(fake_mvtec, CATEGORIES)
    assert not df["path"].str.startswith("/").any()
    assert not df["mask_path"].dropna().str.startswith("/").any()
    assert all(Path(p).exists() for p in df["path"])


def test_absolute_root_is_rejected(fake_mvtec: Path) -> None:
    with pytest.raises(ValueError, match="relative"):
        build_manifest(fake_mvtec.resolve(), CATEGORIES)


def test_size_and_mode_are_recorded(fake_mvtec: Path) -> None:
    df = build_manifest(fake_mvtec, CATEGORIES)
    assert (df["width"] == 64).all()
    assert (df["height"] == 64).all()
    assert set(df.loc[df["category"] == "bottle", "mode"]) == {"RGB"}
    assert set(df.loc[df["category"] == "screw", "mode"]) == {"L"}


def test_sha1_matches_the_file_and_is_unique(fake_mvtec: Path) -> None:
    df = build_manifest(fake_mvtec, CATEGORIES)
    row = df.iloc[0]
    assert row["sha1"] == hashlib.sha1(Path(str(row["path"])).read_bytes()).hexdigest()
    assert df["sha1"].is_unique


def test_output_is_deterministic(fake_mvtec: Path) -> None:
    first = build_manifest(fake_mvtec, CATEGORIES)
    second = build_manifest(fake_mvtec, CATEGORIES)
    assert first.equals(second)
    for _, group in first.groupby("category"):
        assert list(group["path"]) == sorted(group["path"])
