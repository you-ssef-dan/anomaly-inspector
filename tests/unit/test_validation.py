from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandera.errors import SchemaError
from PIL import Image

from anomaly_inspector.data.manifest import build_manifest
from anomaly_inspector.data.validation import manifest_schema, mask_problems


@pytest.fixture
def manifest(fake_mvtec: Path) -> pd.DataFrame:
    return build_manifest(fake_mvtec, ["bottle", "screw"])


def first_defective_mask(manifest: pd.DataFrame) -> Path:
    return Path(str(manifest.loc[manifest["label"] == 1, "mask_path"].iloc[0]))


def test_valid_manifest_passes(manifest: pd.DataFrame) -> None:
    manifest_schema.validate(manifest)


def test_defective_image_without_mask_fails(manifest: pd.DataFrame) -> None:
    defective = manifest.index[manifest["label"] == 1][0]
    manifest.loc[defective, "mask_path"] = None
    with pytest.raises(SchemaError, match="every defective image needs a mask"):
        manifest_schema.validate(manifest)


def test_defect_in_train_fails(manifest: pd.DataFrame) -> None:
    train_row = manifest.index[manifest["source_split"] == "train"][0]
    manifest.loc[train_row, ["label", "mask_path"]] = [1, "some_mask.png"]
    with pytest.raises(SchemaError, match="train must contain only good images"):
        manifest_schema.validate(manifest)


@pytest.mark.parametrize("column", ["path", "sha1"])
def test_duplicates_fail(manifest: pd.DataFrame, column: str) -> None:
    manifest.loc[1, column] = manifest.loc[0, column]
    with pytest.raises(SchemaError, match=f"series '{column}' contains duplicate values"):
        manifest_schema.validate(manifest)


@pytest.mark.parametrize(
    ("column", "value", "rule"),
    [
        ("width", 32, "greater_than_or_equal_to"),
        ("label", 2, r"isin\(\[0, 1\]\)"),
        ("source_split", "val", "isin"),
    ],
)
def test_out_of_range_values_fail(
    manifest: pd.DataFrame, column: str, value: object, rule: str
) -> None:
    defective_test_row = manifest.index[manifest["label"] == 1][0]
    manifest.loc[defective_test_row, column] = value
    with pytest.raises(SchemaError, match=rf"Column '{column}' failed.*{rule}"):
        manifest_schema.validate(manifest)


def test_clean_masks_have_no_problems(manifest: pd.DataFrame) -> None:
    assert mask_problems(manifest) == []


def test_mask_with_wrong_size_is_reported(manifest: pd.DataFrame) -> None:
    Image.fromarray(np.zeros((32, 32), dtype=np.uint8)).save(first_defective_mask(manifest))
    problems = mask_problems(manifest)
    assert len(problems) == 1
    assert "mask size (32, 32) != image size (64, 64)" in problems[0]


def test_mask_with_gray_values_is_reported(manifest: pd.DataFrame) -> None:
    path = first_defective_mask(manifest)
    with Image.open(path) as image:
        mask = np.array(image)
    mask[0, 0] = 128  # what bilinear resizing leaves behind
    Image.fromarray(mask).save(path)
    problems = mask_problems(manifest)
    assert len(problems) == 1
    assert "[0, 128, 255]" in problems[0]
