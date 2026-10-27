"""Train PatchCore on one category, score it on holdout, and log the run to MLflow.

Run it from the repo root, with the MLflow server up (make mlflow-up):

  uv run --env-file .env python scripts/train_patchcore.py --category bottle

The run gets the holdout metrics, threshold.json, the ROC and PRO curves, the failure gallery,
and the model itself, registered as a new version of patchcore-<category>.
"""

import argparse
import tempfile
from collections.abc import Sequence
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from anomaly_inspector.config import load_params
from anomaly_inspector.data.dataset import ManifestDataset
from anomaly_inspector.evaluation.evaluate import Predictions, evaluate, load_masks, predict
from anomaly_inspector.evaluation.gallery import draw_gallery, failure_panels
from anomaly_inspector.evaluation.plots import pro_figure, roc_figure
from anomaly_inspector.evaluation.thresholds import cost_threshold, normal_threshold
from anomaly_inspector.models.backbones import build_backbone
from anomaly_inspector.models.patchcore import PatchCore
from anomaly_inspector.tracking import start_run

# The table of every image, with its category, split and label (built by `dvc repro`).
MANIFEST = "data/processed/manifest.parquet"
GALLERY_PANELS = 4  # images per row of the failure gallery


def score(model: PatchCore, rows: pd.DataFrame, size: int) -> Predictions:
    return predict(model, DataLoader(ManifestDataset(rows, size), batch_size=8))


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
    size = params.data.image_size

    # Only this category's images. train = good parts for the memory bank. calib and
    # val_normal = where the threshold is chosen. holdout = the reporting split: no decision
    # was ever made on it, so its numbers are honest.
    manifest = pd.read_parquet(MANIFEST)
    rows = manifest[manifest["category"] == args.category]
    train, calib, val_normal, holdout = (
        rows[rows["split"] == split] for split in ("train", "calib", "val_normal", "holdout")
    )

    # Everything that can fail goes inside the run, so a crash marks the run FAILED.
    with start_run("patchcore", f"patchcore-{args.category}", args.category, params):
        torch.manual_seed(params.patchcore.seed)
        model = PatchCore(build_backbone(params.patchcore.backbone))
        model.fit(
            DataLoader(ManifestDataset(train, size), batch_size=8),
            params.patchcore.coreset_ratio,
            params.patchcore.seed,
        )

        # 1. Metrics, on holdout.
        labels, masks = holdout["label"].to_numpy(), load_masks(holdout, size)
        predictions = score(model, holdout, size)
        metrics = evaluate(labels, predictions, masks)
        metrics["ms_per_image"] = predictions.ms_per_image
        # The "holdout_" prefix says which split the number comes from: later, promotion
        # compares models on exactly these names.
        mlflow.log_metrics({f"holdout_{name}": value for name, value in metrics.items()})
        print(metrics)

        # 2. threshold.json: the decision rule that ships with the model. Chosen on calib
        # (cost-based) and on val_normal (good parts only), never on holdout.
        cfg = params.threshold
        threshold = {
            "threshold": cost_threshold(
                calib["label"].to_numpy(),
                score(model, calib, size).scores,
                cfg.miss_cost,
                cfg.false_reject_cost,
            ),
            "rule": "reject a part when its score >= threshold",
            "chosen_on": "calib",
            "miss_cost": cfg.miss_cost,
            "false_reject_cost": cfg.false_reject_cost,
            # The cold-start alternative, kept for comparison on holdout.
            "normal_threshold": normal_threshold(
                score(model, val_normal, size).scores, cfg.normal_quantile
            ),
            "normal_quantile": cfg.normal_quantile,
        }
        mlflow.log_dict(threshold, "threshold.json")

        # 3. The curves behind image AUROC and AUPRO. log_figure saves a chart as a PNG.
        mlflow.log_figure(roc_figure(labels, predictions.scores), "curves/roc.png")
        mlflow.log_figure(pro_figure(masks, predictions.maps), "curves/pro.png")

        # 4. The failure gallery on holdout: the picture, and the facts behind each panel.
        panel_rows, records = failure_panels(
            holdout, labels, predictions.scores, predictions.maps, masks, GALLERY_PANELS
        )
        scores = predictions.scores
        subtitle = (
            f"PatchCore, holdout split ({len(holdout)} images). Median image score: good "
            f"{np.median(scores[labels == 0]):.2f}, defective "
            f"{np.median(scores[labels == 1]):.2f}. Image AUROC {metrics['image_auroc']:.3f}, "
            f"AUPRO {metrics['aupro']:.3f}. One colour scale for all panels."
        )
        with tempfile.TemporaryDirectory() as tmp:  # log_artifact uploads a file from disk
            picture = draw_gallery(
                panel_rows,
                f"{args.category}: the closest calls",
                subtitle,
                Path(tmp) / "failures.jpg",
            )
            mlflow.log_artifact(str(picture), "gallery")
        mlflow.log_dict({"panels": records}, "gallery/failures.json")

        # 5. The model, registered as a new version of patchcore-<category>. "pickle" because
        # MLflow's default "pt2" format cannot save a model that returns two tensors (map,
        # score). Loading it needs this package installed. From Phase 5 we log ONNX instead.
        mlflow.pytorch.log_model(
            model,
            name="model",
            registered_model_name=f"patchcore-{args.category}",
            serialization_format="pickle",
        )


if __name__ == "__main__":
    main()
