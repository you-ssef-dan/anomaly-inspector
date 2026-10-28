"""Register the pipeline's PatchCore for one category in MLflow, and promote it if it earns it.

  uv run --env-file .env dvc repro          # train and evaluate; each evaluation is an MLflow run
  uv run --env-file .env python scripts/register_model.py --category screw

Registering stays out of dvc.yaml on purpose: stages turn files into files, so re-running one
never creates a registry version. Later, Airflow or CI calls this script.

It refuses when the category's stages are out of date (the model would not match the code and
params), or when no MLflow run evaluated this exact model file (found by its md5). Otherwise it
logs the model into that run, registers it as the next version of patchcore-<category>, and lets
registry.promote decide between challenger and champion. Running it twice registers once.
"""

import argparse
import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import mlflow
import mlflow.pytorch
from mlflow import MlflowClient
from mlflow.entities import Run

from anomaly_inspector.config import load_params
from anomaly_inspector.models.storage import load_patchcore
from anomaly_inspector.registry import promote
from anomaly_inspector.tracking import md5_of, tracking_uri

EXPERIMENT = "patchcore"


class RegistrationError(RuntimeError):
    pass


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


def register(category: str) -> str:
    """Register the category's model and run the promotion. Returns the version number."""
    tracking_uri()  # refuses to go on without a server
    stale = stale_stages(f"evaluate_patchcore@{category}")
    if stale:
        raise RegistrationError(
            f"out of date: {', '.join(stale)}. Run `uv run --env-file .env dvc repro` first, "
            "so the model matches the committed code and params."
        )

    model_file = Path("models/patchcore") / category / "model.pt"
    run = evaluation_run(category, md5_of(model_file))
    client, name = MlflowClient(), f"patchcore-{category}"

    # Already registered from this run: say so and change nothing (the aliases included).
    run_id = run.info.run_id
    for existing in client.search_model_versions(f"name = '{name}' and run_id = '{run_id}'"):
        print(f"{name}: already registered from run {run_id} as version {existing.version}")
        return str(existing.version)

    # The model goes into the run that evaluated it, so the version carries that run's
    # metrics and tags, which is what promotion judges. "pickle": MLflow's default "pt2"
    # format cannot save a model that returns two tensors. From Phase 5 we log ONNX instead.
    with mlflow.start_run(run_id=run_id):
        info = mlflow.pytorch.log_model(
            load_patchcore(model_file).model,
            name="model",
            registered_model_name=name,
            serialization_format="pickle",
        )
    version = str(info.registered_model_version)
    promoted = promote(client, name, version, load_params().promotion)
    print(f"{name} version {version}: {'champion' if promoted else 'challenger only'}")
    return version


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--category", required=True)
    args = parser.parse_args(argv)
    register(args.category)


if __name__ == "__main__":
    main()
