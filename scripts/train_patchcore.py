"""Train PatchCore on one category, score it on holdout, and log the run to MLflow.

Run it from the repo root, with the MLflow server up (make mlflow-up):

  uv run --env-file .env python scripts/train_patchcore.py --category bottle
"""

import argparse

import mlflow
import pandas as pd
import torch
from torch.utils.data import DataLoader

from anomaly_inspector.config import load_params
from anomaly_inspector.data.dataset import ManifestDataset
from anomaly_inspector.evaluation.evaluate import evaluate, load_masks, predict
from anomaly_inspector.models.backbones import build_backbone
from anomaly_inspector.models.patchcore import PatchCore
from anomaly_inspector.tracking import start_run

# The table of every image, with its category, split and label (built by `dvc repro`).
MANIFEST = "data/processed/manifest.parquet"


def main() -> None:
    # Which category to train comes from the command line: --category bottle.
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--category", required=True)
    args = parser.parse_args()

    # Load the settings once. The same object trains the model and is logged by start_run,
    # so the params in MLflow are exactly the ones used.
    params = load_params()
    size = params.data.image_size

    # Only this category's images. train = good parts for the memory bank.
    # holdout = the reporting split: no decision was ever made on it, so its numbers are honest.
    manifest = pd.read_parquet(MANIFEST)
    rows = manifest[manifest["category"] == args.category]
    train = rows[rows["split"] == "train"]
    holdout = rows[rows["split"] == "holdout"]

    # Everything that can fail goes inside the run, so a crash marks the run FAILED.
    with start_run("patchcore", f"patchcore-{args.category}", args.category, params):
        torch.manual_seed(params.patchcore.seed)
        model = PatchCore(build_backbone(params.patchcore.backbone))
        model.fit(
            DataLoader(ManifestDataset(train, size), batch_size=8),
            params.patchcore.coreset_ratio,
            params.patchcore.seed,
        )

        predictions = predict(model, DataLoader(ManifestDataset(holdout, size), batch_size=8))
        metrics = evaluate(holdout["label"].to_numpy(), predictions, load_masks(holdout, size))
        metrics["ms_per_image"] = predictions.ms_per_image

        # The "holdout_" prefix says which split the number comes from: later, promotion
        # compares models on exactly these names.
        mlflow.log_metrics({f"holdout_{name}": value for name, value in metrics.items()})
        print(metrics)


if __name__ == "__main__":
    main()
