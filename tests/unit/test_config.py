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
    assert params.stfpm.epochs == 100


def test_defaults_fill_missing_values(tmp_path: Path) -> None:
    params = load_params(write_params(tmp_path, MINIMAL))
    assert params.data.image_size == 256
    assert params.data.val_normal_fraction == 0.1
    assert params.data.seed == 42


def test_stfpm_defaults_are_the_papers_settings(tmp_path: Path) -> None:
    stfpm = load_params(write_params(tmp_path, MINIMAL)).stfpm  # section left out entirely
    assert (stfpm.epochs, stfpm.lr, stfpm.batch_size) == (100, 0.4, 32)
    assert (stfpm.momentum, stfpm.weight_decay, stfpm.patience) == (0.9, 1e-4, 10)


@pytest.mark.parametrize(
    ("key", "value"),
    [("epochs", 0), ("lr", 0), ("batch_size", 0), ("momentum", 1.0), ("patience", 0)],
)
def test_impossible_stfpm_settings_are_rejected(tmp_path: Path, key: str, value: float) -> None:
    content = deepcopy(MINIMAL) | {"stfpm": {key: value}}
    with pytest.raises(ValidationError, match=key):
        load_params(write_params(tmp_path, content))


def test_the_threshold_defaults_are_a_10_to_1_cost_and_the_99th_percentile(
    tmp_path: Path,
) -> None:
    threshold = load_params(write_params(tmp_path, MINIMAL)).threshold
    assert (threshold.miss_cost, threshold.false_reject_cost) == (10, 1)
    assert threshold.normal_quantile == 0.99


@pytest.mark.parametrize(
    ("key", "value"),
    [("miss_cost", 0), ("false_reject_cost", -1), ("normal_quantile", 1.0)],
)
def test_impossible_threshold_settings_are_rejected(tmp_path: Path, key: str, value: float) -> None:
    content = deepcopy(MINIMAL) | {"threshold": {key: value}}
    with pytest.raises(ValidationError, match=key):
        load_params(write_params(tmp_path, content))


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    content = deepcopy(MINIMAL)
    content["data"]["image_sise"] = 256  # typo for image_size
    with pytest.raises(ValidationError, match="image_sise"):
        load_params(write_params(tmp_path, content))


def test_section_without_a_model_is_rejected(tmp_path: Path) -> None:
    content = deepcopy(MINIMAL)
    # Phase 5 adds this section together with its class
    content["onnx"] = {"opset": 18}
    with pytest.raises(ValidationError, match="onnx"):
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


def test_the_promotion_defaults_are_a_200_ms_budget_and_a_small_aupro_tolerance(
    tmp_path: Path,
) -> None:
    promotion = load_params(write_params(tmp_path, MINIMAL)).promotion
    assert (promotion.latency_budget_ms, promotion.aupro_tolerance) == (200, 0.005)


def test_the_export_defaults_calibrate_on_100_images_and_time_on_4_threads(
    tmp_path: Path,
) -> None:
    export = load_params(write_params(tmp_path, MINIMAL)).export
    assert (export.calibration_images, export.threads) == (100, 4)
