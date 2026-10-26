import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml
from pandera.errors import SchemaErrors
from PIL import Image

from anomaly_inspector.config import load_params
from anomaly_inspector.data.manifest import COLUMNS
from anomaly_inspector.data.splits import SPLITS
from anomaly_inspector.pipelines.prepare import MANIFEST_PATH, SUMMARY_PATH, main, prepare


@pytest.fixture
def params_file(fake_mvtec: Path) -> Path:
    """params.yaml for the fake bottle dataset, written where the stage runs (the temp folder)."""
    content: dict[str, Any] = {
        "data": {"root": fake_mvtec.as_posix(), "categories": ["bottle"], "seed": 42}
    }
    path = Path("params.yaml")
    path.write_text(yaml.safe_dump(content))
    return path


def a_defect_mask(fake_mvtec: Path) -> Path:
    return next((fake_mvtec / "bottle" / "ground_truth" / "broken_large").glob("*_mask.png"))


def test_writes_the_manifest_with_a_split_column(params_file: Path) -> None:
    manifest = prepare(load_params(params_file))
    written = pd.read_parquet(MANIFEST_PATH)
    assert list(written.columns) == [*COLUMNS, "split"]
    assert len(written) == 8
    assert set(written["split"]) == set(SPLITS)
    assert list(written["path"]) == list(manifest["path"])


def test_summary_counts_per_category_split_and_defect_type(params_file: Path) -> None:
    prepare(load_params(params_file))
    summary = json.loads(SUMMARY_PATH.read_text())
    assert summary["images"] == 8
    assert summary["per_category"] == {"bottle": 8}
    assert summary["per_category_split"] == {
        "bottle": {"train": 3, "val_normal": 1, "calib": 2, "holdout": 2}
    }
    assert summary["per_category_defect_type"] == {"bottle": {"broken_large": 2, "good": 6}}


def test_running_twice_gives_byte_identical_files(params_file: Path) -> None:
    params = load_params(params_file)
    prepare(params)
    first = (MANIFEST_PATH.read_bytes(), SUMMARY_PATH.read_bytes())
    prepare(params)
    assert (MANIFEST_PATH.read_bytes(), SUMMARY_PATH.read_bytes()) == first


def test_the_seed_in_params_decides_the_split(params_file: Path) -> None:
    def splits_for(seed: int) -> list[str]:
        content = yaml.safe_load(params_file.read_text())
        content["data"]["seed"] = seed
        params_file.write_text(yaml.safe_dump(content))
        return list(prepare(load_params(params_file))["split"])

    assert splits_for(42) == splits_for(42)
    assert splits_for(42) != splits_for(7)


def test_a_bad_mask_stops_the_stage_and_writes_nothing(params_file: Path, fake_mvtec: Path) -> None:
    Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(a_defect_mask(fake_mvtec))
    with pytest.raises(ValueError, match="Invalid masks"):
        prepare(load_params(params_file))
    assert not MANIFEST_PATH.exists()
    assert not SUMMARY_PATH.exists()


def test_a_missing_mask_stops_the_stage_and_writes_nothing(
    params_file: Path, fake_mvtec: Path
) -> None:
    a_defect_mask(fake_mvtec).unlink()
    with pytest.raises(SchemaErrors) as error:
        prepare(load_params(params_file))
    assert "every defective image needs a mask" in set(error.value.failure_cases["check"])
    assert not MANIFEST_PATH.exists()
    assert not SUMMARY_PATH.exists()


def test_main_reads_params_yaml_from_the_working_directory(
    params_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    main()
    assert "Wrote 8 rows" in capsys.readouterr().out
    assert MANIFEST_PATH.exists()


def test_module_runs_the_way_dvc_runs_it(params_file: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "anomaly_inspector.pipelines.prepare"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Wrote 8 rows" in result.stdout
    assert MANIFEST_PATH.exists()
    assert SUMMARY_PATH.exists()
