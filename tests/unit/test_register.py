"""pipelines/register.py on the fake project: the pipeline's ONNX model goes to the registry, once.

The fake project has no DVC repository, so stale_stages is replaced in most tests; the last
test runs it against a real one.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import mlflow
import mlflow.onnx
import onnx
import pandas as pd
import pytest
import yaml
from torch.utils.data import DataLoader

from anomaly_inspector.data.dataset import ManifestDataset
from anomaly_inspector.evaluation.evaluate import predict
from anomaly_inspector.evaluation.metrics import image_auroc
from anomaly_inspector.export.onnx_export import OnnxPatchCore, read_metadata
from anomaly_inspector.pipelines import evaluate_patchcore, export_patchcore, train_patchcore
from anomaly_inspector.pipelines import register as registration


@pytest.fixture(scope="module")
def script() -> ModuleType:
    return registration


@pytest.fixture
def evaluated(fake_project: Path, monkeypatch: pytest.MonkeyPatch, script: ModuleType) -> Path:
    """The pipeline has run with a server: the evaluation is a run, and model.onnx exists."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{fake_project / 'store.db'}")
    monkeypatch.setattr(script, "stale_stages", lambda target: [])
    for stage in (train_patchcore, evaluate_patchcore, export_patchcore):
        stage.main(["--category", "bottle"])
    # Promotion also judges speed, and a busy CI runner once took 245 ms per fake image, over the
    # 200 ms budget. These tests are about registering, so the budget is out of the way. Changed
    # after the stages: the evaluation run must come from a clean git tree to be promotable.
    params = yaml.safe_load(Path("params.yaml").read_text())
    params["promotion"] = {"latency_budget_ms": 60_000}
    Path("params.yaml").write_text(yaml.safe_dump(params))
    return fake_project


def the_evaluation() -> mlflow.entities.Run:
    (run,) = mlflow.search_runs(experiment_names=["patchcore"], output_format="list")
    return run


def test_the_onnx_model_joins_the_run_that_evaluated_it_and_becomes_champion(
    script: ModuleType, evaluated: Path, tmp_path: Path
) -> None:
    assert script.register("bottle") == ("1", True)

    client = mlflow.MlflowClient()
    run = the_evaluation()
    version = client.get_model_version("patchcore-bottle", "1")
    assert version.run_id == run.info.run_id
    assert version.tags == {"format": "onnx"}
    assert client.get_registered_model("patchcore-bottle").aliases == {
        "champion": 1,
        "challenger": 1,
    }
    exported = json.loads(Path("reports/patchcore/bottle/export.json").read_text())
    logged = mlflow.get_run(run.info.run_id).data.metrics
    assert logged["onnx_image_auroc"] == pytest.approx(exported["onnx_image_auroc"])

    # The champion, loaded fresh from the store: the ONNX file with its metadata, and it gives
    # back the logged holdout AUROC.
    path = tmp_path / "champion.onnx"
    onnx.save(mlflow.onnx.load_model("models:/patchcore-bottle@champion"), str(path))
    assert read_metadata(path) == read_metadata(export_patchcore.onnx_path("bottle"))
    rows = pd.read_parquet("data/processed/manifest.parquet")
    holdout = rows[(rows["category"] == "bottle") & (rows["split"] == "holdout")]
    loader = DataLoader(ManifestDataset(holdout, 64), batch_size=8)
    scores = predict(OnnxPatchCore(path), loader).scores
    assert image_auroc(holdout["label"].to_numpy(), scores) == pytest.approx(
        logged["holdout_image_auroc"]
    )


def test_an_older_pytorch_version_of_the_same_run_does_not_count(
    script: ModuleType, evaluated: Path
) -> None:
    # Phase 3 and 4 registered the PyTorch model from the same runs, with no format tag.
    client, run = mlflow.MlflowClient(), the_evaluation()
    client.create_registered_model("patchcore-bottle")
    client.create_model_version("patchcore-bottle", f"{run.info.artifact_uri}/old", run.info.run_id)

    assert script.register("bottle").version == "2"
    assert client.get_model_version("patchcore-bottle", "2").tags == {"format": "onnx"}
    assert script.register("bottle") == ("2", False)  # and only once: no new promotion


def test_the_pipeline_must_be_fresh_up_to_the_export(
    script: ModuleType, evaluated: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asked: list[str] = []

    def stale_stages(target: str) -> list[str]:
        asked.append(target)
        return []

    monkeypatch.setattr(script, "stale_stages", stale_stages)
    script.register("bottle")
    assert asked == ["export_patchcore@bottle"]


def test_registering_twice_creates_one_version(script: ModuleType, evaluated: Path) -> None:
    assert script.register("bottle") == ("1", True)
    assert script.register("bottle") == ("1", False)
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


def test_the_last_line_says_which_categories_have_a_new_champion(
    script: ModuleType,
    fake_project: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    outcomes = {"bottle": script.Registration("3", True), "screw": script.Registration("1", False)}
    asked: list[str] = []

    def register(category: str) -> object:
        asked.append(category)
        return outcomes[category]

    monkeypatch.setattr(script, "register", register)
    script.main(["--all"])
    assert asked == ["bottle", "screw"]  # every category in params.yaml, in its order
    last = capsys.readouterr().out.splitlines()[-1]
    assert json.loads(last) == {"bottle": True, "screw": False}

    script.main(["--category", "screw"])
    assert json.loads(capsys.readouterr().out.splitlines()[-1]) == {"screw": False}


@pytest.mark.parametrize("args", [[], ["--all", "--category", "bottle"]])
def test_categories_are_named_or_all_of_them(script: ModuleType, args: list[str]) -> None:
    with pytest.raises(SystemExit):
        script.main(args)
