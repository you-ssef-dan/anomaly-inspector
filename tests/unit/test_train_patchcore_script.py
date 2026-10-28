"""train_patchcore.py end to end on the fake dataset: tiny untrained backbone, file MLflow store.

No download, no Docker. It proves the run gets its metrics, artifacts and model before the
script ever meets the real data.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import mlflow
import mlflow.pytorch
import pandas as pd
import pytest

from anomaly_inspector.evaluation.metrics import image_auroc
from anomaly_inspector.pipelines.evaluate_patchcore import score

SCRIPT = Path(__file__).parents[2] / "scripts" / "train_patchcore.py"


@pytest.fixture(scope="module")
def script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("train_patchcore_script", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["train_patchcore_script"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def project(fake_project: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{fake_project / 'mlflow.db'}")
    # Sample system metrics every half second instead of 10, so this short run gets some.
    monkeypatch.setenv("MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL", "0.5")
    return fake_project


def the_run() -> Any:
    (run,) = mlflow.search_runs(experiment_names=["patchcore"], output_format="list")
    return run


def test_the_run_holds_the_metrics_the_artifacts_and_a_model_that_reloads(
    script: ModuleType, project: Path
) -> None:
    script.main(["--category", "bottle"])

    run = the_run()
    assert run.info.status == "FINISHED"
    assert run.info.run_name == "patchcore-bottle"
    model_metrics = {name for name in run.data.metrics if not name.startswith("system/")}
    assert model_metrics == {
        "holdout_image_auroc",
        "holdout_pixel_auroc",
        "holdout_aupro",
        "holdout_ms_per_image",
    }
    # CPU and RAM were sampled while the run was open.
    assert {"system/cpu_utilization_percentage", "system/system_memory_usage_megabytes"} <= set(
        run.data.metrics
    )
    files = {a.path for a in mlflow.artifacts.list_artifacts(run_id=run.info.run_id)}
    assert files == {"threshold.json", "figures"}  # the model is stored apart (MLflow 3)
    figures = mlflow.artifacts.list_artifacts(run_id=run.info.run_id, artifact_path="figures")
    assert {a.path for a in figures} == {
        "figures/roc.png",
        "figures/pro.png",
        "figures/failures.jpg",
        "figures/failures.json",
    }

    (logged_model,) = mlflow.search_logged_models(
        filter_string=f"source_run_id = '{run.info.run_id}'", output_format="list"
    )
    assert logged_model.name == "model"

    threshold = mlflow.artifacts.load_dict(f"runs:/{run.info.run_id}/threshold.json")
    assert threshold["chosen_on"] == "calib"
    assert (threshold["miss_cost"], threshold["false_reject_cost"]) == (10, 1)

    # The first clean version of a category is both challenger and champion, and the
    # champion, loaded fresh from the store, gives back the logged holdout AUROC.
    client = mlflow.MlflowClient()
    assert client.get_model_version("patchcore-bottle", "1").run_id == run.info.run_id
    assert client.get_registered_model("patchcore-bottle").aliases == {
        "champion": 1,
        "challenger": 1,
    }
    model = mlflow.pytorch.load_model("models:/patchcore-bottle@champion")
    manifest = pd.read_parquet("data/processed/manifest.parquet")
    holdout = manifest[(manifest["category"] == "bottle") & (manifest["split"] == "holdout")]
    reloaded = score(model, holdout, 64).scores
    assert image_auroc(holdout["label"].to_numpy(), reloaded) == pytest.approx(
        run.data.metrics["holdout_image_auroc"]
    )
