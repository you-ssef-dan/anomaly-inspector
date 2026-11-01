"""Register the pipeline's PatchCore in MLflow, category by category, and promote it if it earns it.

  uv run --env-file .env dvc repro          # train and evaluate; each evaluation is an MLflow run
  uv run --env-file .env python -m anomaly_inspector.pipelines.register --category screw
  uv run --env-file .env python -m anomaly_inspector.pipelines.register --all

Registering stays out of dvc.yaml on purpose: stages turn files into files, so re-running one
never creates a registry version. The retrain_patchcore DAG calls this module; its last line,
like {"bottle": true}, says which categories have a new champion.

It registers model.onnx, the file the API serves, with its threshold and image size inside. It
refuses when the category's stages are out of date (the model would not match the code and
params), or when no MLflow run evaluated this model (found by the md5 of model.pt, which the ONNX
file was exported from). Otherwise it logs the ONNX model into that run, registers it as the next
version of patchcore-<category>, and lets registry.promote decide between challenger and
champion. Running it twice registers once.
"""

import argparse
import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

import mlflow
import mlflow.onnx
import onnx
from mlflow import MlflowClient
from mlflow.entities import Run

from anomaly_inspector.config import load_params
from anomaly_inspector.registry import promote
from anomaly_inspector.tracking import md5_of, tracking_uri

EXPERIMENT = "patchcore"
FORMAT_TAG = ("format", "onnx")  # on every version this script registers


class RegistrationError(RuntimeError):
    pass


class Registration(NamedTuple):
    version: str
    promoted: bool  # this call made it the champion


def stale_stages(target: str) -> list[str]:
    """The stages DVC would re-run to bring target up to date, its upstream stages included."""
    out = subprocess.run(
        [sys.executable, "-m", "dvc", "status", "--json", "--with-deps", target],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return sorted(json.loads(out))


def evaluation_run(category: str, model_md5: str) -> Run:
    """The latest finished MLflow run that evaluated this exact model file."""
    runs = mlflow.search_runs(
        experiment_names=[EXPERIMENT],
        filter_string=(
            f"tags.category = '{category}' and tags.model_md5 = '{model_md5}' "
            "and attributes.status = 'FINISHED'"
        ),
        order_by=["attributes.start_time DESC"],
        max_results=1,
        output_format="list",
    )
    if not runs:
        raise RegistrationError(
            f"no MLflow run evaluated this {category} model. Evaluate it with the server set: "
            f"uv run --env-file .env dvc repro --force evaluate_patchcore@{category}"
        )
    return runs[0]


def register(category: str) -> Registration:
    """Register the category's model and run the promotion."""
    tracking_uri()  # refuses to go on without a server
    stale = stale_stages(f"export_patchcore@{category}")
    if stale:
        raise RegistrationError(
            f"out of date: {', '.join(stale)}. Run `uv run --env-file .env dvc repro` first, "
            "so the model matches the committed code and params."
        )

    models, reports = Path("models/patchcore") / category, Path("reports/patchcore") / category
    run = evaluation_run(category, md5_of(models / "model.pt"))
    client, name = MlflowClient(), f"patchcore-{category}"

    # This run's ONNX model is already registered: say so and change nothing (aliases included).
    # Older versions of the same run hold the PyTorch model; they do not count.
    run_id = run.info.run_id
    for existing in client.search_model_versions(f"name = '{name}' and run_id = '{run_id}'"):
        if existing.tags.get(FORMAT_TAG[0]) == FORMAT_TAG[1]:
            print(f"{name}: already registered from run {run_id} as version {existing.version}")
            return Registration(str(existing.version), promoted=False)

    # The model goes into the run that evaluated it, so the version carries that run's
    # metrics and tags, which is what promotion judges. The export stage's check (ONNX and
    # PyTorch agree on holdout) goes with it.
    exported = json.loads((reports / "export.json").read_text())
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics({k: v for k, v in exported.items() if k.startswith("onnx_")})
        info = mlflow.onnx.log_model(
            onnx.load(str(models / "model.onnx")),
            name="model",
            registered_model_name=name,
            save_as_external_data=False,  # one file, as exported
        )
    version = str(info.registered_model_version)
    client.set_model_version_tag(name, version, *FORMAT_TAG)
    promoted = promote(client, name, version, load_params().promotion)
    print(f"{name} version {version}: {'champion' if promoted else 'challenger only'}")
    return Registration(version, promoted)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--category", nargs="+", help="one or more categories")
    which.add_argument("--all", action="store_true", help="every category in params.yaml")
    args = parser.parse_args(argv)
    categories = load_params().data.categories if args.all else args.category
    promoted = {category: register(category).promoted for category in categories}
    print(json.dumps(promoted))  # the last line: what a caller (the DAG) reads


if __name__ == "__main__":
    main()
