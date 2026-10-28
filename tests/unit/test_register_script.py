"""register_model.py on the fake project: the pipeline's model goes to the registry, once.

The fake project has no DVC repository, so stale_stages is replaced in most tests; the last
test runs it against a real one.
"""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import mlflow
import mlflow.pytorch
import pandas as pd
import pytest
import yaml

from anomaly_inspector.evaluation.metrics import image_auroc
from anomaly_inspector.pipelines import evaluate_patchcore, train_patchcore

SCRIPT = Path(__file__).parents[2] / "scripts" / "register_model.py"


@pytest.fixture(scope="module")
def script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("register_model", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["register_model"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def evaluated(fake_project: Path, monkeypatch: pytest.MonkeyPatch, script: ModuleType) -> Path:
    """The pipeline has run with a server: model.pt exists and its evaluation is a run."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{fake_project / 'store.db'}")
    monkeypatch.setattr(script, "stale_stages", lambda target: [])
    train_patchcore.main(["--category", "bottle"])
    evaluate_patchcore.main(["--category", "bottle"])
    return fake_project


def the_evaluation() -> mlflow.entities.Run:
    (run,) = mlflow.search_runs(experiment_names=["patchcore"], output_format="list")
    return run


def test_the_model_joins_the_run_that_evaluated_it_and_becomes_champion(
    script: ModuleType, evaluated: Path
) -> None:
    assert script.register("bottle") == "1"

    client = mlflow.MlflowClient()
    run = the_evaluation()
    assert client.get_model_version("patchcore-bottle", "1").run_id == run.info.run_id
    assert client.get_registered_model("patchcore-bottle").aliases == {
        "champion": 1,
        "challenger": 1,
    }
    # The champion, loaded fresh from the store, gives back the logged holdout AUROC.
    model = mlflow.pytorch.load_model("models:/patchcore-bottle@champion")
    rows = pd.read_parquet("data/processed/manifest.parquet")
    holdout = rows[(rows["category"] == "bottle") & (rows["split"] == "holdout")]
    scores = evaluate_patchcore.score(model, holdout, 64).scores
    assert image_auroc(holdout["label"].to_numpy(), scores) == pytest.approx(
        run.data.metrics["holdout_image_auroc"]
    )


def test_registering_twice_creates_one_version(script: ModuleType, evaluated: Path) -> None:
    assert script.register("bottle") == "1"
    assert script.register("bottle") == "1"
    versions = mlflow.MlflowClient().search_model_versions("name = 'patchcore-bottle'")
    assert len(versions) == 1


def test_an_out_of_date_pipeline_is_refused(
    script: ModuleType, evaluated: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(script, "stale_stages", lambda target: ["train_patchcore@bottle"])
    with pytest.raises(script.RegistrationError, match="train_patchcore@bottle"):
        script.register("bottle")
    assert mlflow.MlflowClient().search_registered_models() == []


def test_a_model_evaluated_without_a_server_is_refused(
    script: ModuleType, evaluated: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A new setting, retrained and evaluated with no server: no run knows the new model.
    params = yaml.safe_load(Path("params.yaml").read_text())
    params["patchcore"]["coreset_ratio"] = 0.25
    Path("params.yaml").write_text(yaml.safe_dump(params))
    uri = os.environ["MLFLOW_TRACKING_URI"]
    monkeypatch.delenv("MLFLOW_TRACKING_URI")
    train_patchcore.main(["--category", "bottle"])
    evaluate_patchcore.main(["--category", "bottle"])

    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    with pytest.raises(
        script.RegistrationError, match="dvc repro --force evaluate_patchcore@bottle"
    ):
        script.register("bottle")
    assert mlflow.MlflowClient().search_registered_models() == []


def test_without_a_server_nothing_is_registered(
    script: ModuleType, evaluated: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("MLFLOW_TRACKING_URI")
    with pytest.raises(RuntimeError, match="MLFLOW_TRACKING_URI"):
        script.register("bottle")


def test_stale_stages_lists_what_dvc_would_rerun_upstream_included(
    script: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@example.com"]
    dvc = [sys.executable, "-m", "dvc"]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*dvc, "init", "-q"], check=True)
    Path("input.txt").write_text("1\n")
    Path("dvc.yaml").write_text(
        "stages:\n"
        "  first:\n    cmd: cp input.txt middle.txt\n    deps: [input.txt]\n"
        "    outs: [middle.txt]\n"
        "  second:\n    cmd: cp middle.txt last.txt\n    deps: [middle.txt]\n"
        "    outs: [last.txt]\n"
    )
    subprocess.run([*dvc, "repro", "-q"], check=True)
    assert script.stale_stages("second") == []

    Path("input.txt").write_text("2\n")
    assert script.stale_stages("second") == ["first"]
