"""Train PatchCore on one category, score it on holdout, and log the run to MLflow.

Run it from the repo root, with the MLflow server up (make mlflow-up):

  uv run --env-file .env python scripts/train_patchcore.py --category bottle

It does in one go what the DVC stages train_patchcore and evaluate_patchcore do, with the same
code, and logs the result: the holdout metrics, threshold.json, the curves, the failure gallery,
and the model itself, registered as a new version of patchcore-<category>. That version becomes
the challenger, and the champion too if registry.should_promote accepts it.
"""

import argparse
import tempfile
from collections.abc import Sequence
from pathlib import Path

import mlflow
import mlflow.pytorch
import pandas as pd
from mlflow import MlflowClient

from anomaly_inspector.config import load_params
from anomaly_inspector.pipelines.evaluate_patchcore import Outputs, evaluate_patchcore
from anomaly_inspector.pipelines.train_patchcore import train_patchcore
from anomaly_inspector.registry import promote
from anomaly_inspector.tracking import start_run

# The table of every image, with its category, split and label (built by `dvc repro`).
MANIFEST = "data/processed/manifest.parquet"


def main(argv: Sequence[str] | None = None) -> None:
    # Which category to train comes from the command line: --category bottle.
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--category", required=True)
    args = parser.parse_args(argv)

    # Load the settings once. The same object trains the model and is logged by start_run,
    # so the params in MLflow are exactly the ones used.
    params = load_params()

    # Only this category's images. train = good parts for the memory bank. calib and
    # val_normal = where the threshold is chosen. holdout = the reporting split: no decision
    # was ever made on it, so its numbers are honest.
    manifest = pd.read_parquet(MANIFEST)
    rows = manifest[manifest["category"] == args.category]

    # Everything that can fail goes inside the run, so a crash marks the run FAILED.
    with start_run("patchcore", f"patchcore-{args.category}", args.category, params):
        model = train_patchcore(params, rows[rows["split"] == "train"])

        # The evaluate stage's outputs go to a temporary folder, then to the run.
        with tempfile.TemporaryDirectory() as tmp:
            out = Outputs.under(Path(tmp), Path(tmp))
            metrics = evaluate_patchcore(
                model,
                params.data.image_size,
                rows,
                params.threshold,
                out,
                f"{args.category}: the closest calls",
            )
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(str(out.threshold))
            mlflow.log_artifacts(str(out.figures), "figures")
        print(metrics)

        # The model, registered as a new version of patchcore-<category>. "pickle" because
        # MLflow's default "pt2" format cannot save a model that returns two tensors (map,
        # score). Loading it needs this package installed. From Phase 5 we log ONNX instead.
        name = f"patchcore-{args.category}"
        info = mlflow.pytorch.log_model(
            model, name="model", registered_model_name=name, serialization_format="pickle"
        )

        # The new version is the challenger; tested rules decide if it becomes champion.
        version = str(info.registered_model_version)
        promoted = promote(MlflowClient(), name, version, params.promotion)
        print(f"{name} version {version}: {'champion' if promoted else 'challenger only'}")


if __name__ == "__main__":
    main()
