from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from anomaly_inspector.config import load_params

REPO_ROOT = Path(__file__).parents[2]
MINIMAL: dict[str, Any] = {"data": {"root": "data/raw/mvtec_ad", "categories": ["bottle"]}}


def write_params(tmp_path: Path, content: dict[str, Any]) -> Path:
    path = tmp_path / "params.yaml"
    path.write_text(yaml.safe_dump(content))
    return path


def test_repo_params_file_is_valid() -> None:
    params = load_params(REPO_ROOT / "params.yaml")
    assert params.data.root == Path("data/raw/mvtec_ad")
    assert params.data.categories == ["bottle", "carpet", "screw"]


def test_defaults_fill_missing_values(tmp_path: Path) -> None:
    params = load_params(write_params(tmp_path, MINIMAL))
    assert params.data.image_size == 256
    assert params.data.val_normal_fraction == 0.1
    assert params.data.seed == 42


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    content = deepcopy(MINIMAL)
    content["data"]["image_sise"] = 256  # typo for image_size
    with pytest.raises(ValidationError, match="image_sise"):
        load_params(write_params(tmp_path, content))


def test_section_without_a_model_is_rejected(tmp_path: Path) -> None:
    content = deepcopy(MINIMAL)
    # Phase 2 adds this section together with its class
    content["patchcore"] = {"coreset_ratio": 0.01}
    with pytest.raises(ValidationError, match="patchcore"):
        load_params(write_params(tmp_path, content))


@pytest.mark.parametrize(
    ("key", "value"),
    [("image_size", 32), ("val_normal_fraction", 1.5), ("val_normal_fraction", 0)],
)
def test_out_of_range_value_is_rejected(tmp_path: Path, key: str, value: float) -> None:
    content = deepcopy(MINIMAL)
    content["data"][key] = value
    with pytest.raises(ValidationError, match=key):
        load_params(write_params(tmp_path, content))


def test_params_are_frozen(tmp_path: Path) -> None:
    params = load_params(write_params(tmp_path, MINIMAL))
    with pytest.raises(ValidationError):
        params.data.seed = 0
