"""DVC stage evaluate_patchcore@<category>: score the saved model and write what explains it.

  python -m anomaly_inspector.pipelines.evaluate_patchcore --category bottle

Reads models/patchcore/<category>/model.pt and writes:
  models/patchcore/<category>/threshold.json   the decision rule that ships with the model
  reports/patchcore/<category>/metrics.json    holdout metrics (dvc metrics show / diff)
  reports/patchcore/<category>/roc.csv         the curves as points (dvc plots show / diff)
  reports/patchcore/<category>/pro.csv
  reports/patchcore/<category>/figures/        roc.png, pro.png, failures.jpg, failures.json

Nothing is fitted here. The threshold is chosen on calib and val_normal; holdout is only scored,
so its numbers stay honest.

With MLFLOW_TRACKING_URI set, the evaluation is also an MLflow run in the "patchcore" experiment,
tagged with the md5 of the model it scored. The model itself is not logged or registered here:
scripts/register_model.py does that, so re-running this stage never adds a registry version.
"""

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from torch.utils.data import DataLoader

from anomaly_inspector.config import ThresholdConfig, load_params
from anomaly_inspector.data.dataset import ManifestDataset
from anomaly_inspector.evaluation.evaluate import Predictions, evaluate, load_masks, predict
from anomaly_inspector.evaluation.gallery import draw_gallery, failure_panels
from anomaly_inspector.evaluation.plots import pro_figure, pro_points, roc_figure, roc_points
from anomaly_inspector.evaluation.thresholds import cost_threshold, normal_threshold
from anomaly_inspector.models.patchcore import PatchCore
from anomaly_inspector.models.storage import load_patchcore
from anomaly_inspector.tracking import md5_of, start_run, tracking_enabled

MANIFEST_PATH = Path("data/processed/manifest.parquet")
GALLERY_PANELS = 4  # images per row of the failure gallery


@dataclass(frozen=True)
class Outputs:
    """Where one category's evaluation goes. dvc.yaml lists the same paths."""

    threshold: Path
    metrics: Path
    roc: Path
    pro: Path
    figures: Path

    @classmethod
    def under(cls, models: Path, reports: Path) -> "Outputs":
        return cls(
            threshold=models / "threshold.json",
            metrics=reports / "metrics.json",
            roc=reports / "roc.csv",
            pro=reports / "pro.csv",
            figures=reports / "figures",
        )

    @classmethod
    def for_category(cls, category: str) -> "Outputs":
        return cls.under(Path("models/patchcore") / category, Path("reports/patchcore") / category)


def score(model: PatchCore, rows: pd.DataFrame, size: int) -> Predictions:
    return predict(model, DataLoader(ManifestDataset(rows, size), batch_size=8))


def choose_threshold(
    model: PatchCore, calib: pd.DataFrame, val_normal: pd.DataFrame, size: int, cfg: ThresholdConfig
) -> dict[str, Any]:
    """threshold.json: the cost-based threshold (calib) and the good-parts-only one (val_normal)."""
    return {
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


def write_curve(path: Path, x: ArrayLike, y: ArrayLike, names: tuple[str, str]) -> None:
    # Six significant digits: enough to draw, and small diffs in git.
    frame = pd.DataFrame({names[0]: np.asarray(x), names[1]: np.asarray(y)})
    frame.to_csv(path, index=False, float_format="%.6g")


def write_json(path: Path, content: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2, sort_keys=True) + "\n")


def evaluate_patchcore(
    model: PatchCore, size: int, rows: pd.DataFrame, cfg: ThresholdConfig, out: Outputs, title: str
) -> dict[str, float]:
    """Score one category's model and write every output. Returns the holdout metrics."""
    split = {name: rows[rows["split"] == name] for name in ("calib", "val_normal", "holdout")}
    holdout = split["holdout"]
    labels, masks = holdout["label"].to_numpy(), load_masks(holdout, size)
    predictions = score(model, holdout, size)

    # 1. Metrics, on holdout. The "holdout_" prefix says which split they come from, and they
    # keep the same names in MLflow, where promotion compares them.
    found = evaluate(labels, predictions, masks) | {"ms_per_image": predictions.ms_per_image}
    metrics = {f"holdout_{name}": value for name, value in found.items()}
    write_json(out.metrics, metrics)

    # 2. threshold.json, next to the model it belongs to.
    threshold = choose_threshold(model, split["calib"], split["val_normal"], size, cfg)
    write_json(out.threshold, threshold)

    # 3. The curves: points for dvc plots, pictures for people.
    out.roc.parent.mkdir(parents=True, exist_ok=True)
    write_curve(out.roc, *roc_points(labels, predictions.scores), names=("fpr", "tpr"))
    write_curve(out.pro, *pro_points(masks, predictions.maps), names=("fpr", "pro"))
    out.figures.mkdir(parents=True, exist_ok=True)
    roc_figure(labels, predictions.scores).savefig(out.figures / "roc.png")
    pro_figure(masks, predictions.maps).savefig(out.figures / "pro.png")

    # 4. The failure gallery on holdout: the picture, and the facts behind each panel.
    scores = predictions.scores
    panel_rows, records = failure_panels(
        holdout, labels, scores, predictions.maps, masks, GALLERY_PANELS
    )
    subtitle = (
        f"PatchCore, holdout split ({len(holdout)} images). Median image score: good "
        f"{np.median(scores[labels == 0]):.2f}, defective {np.median(scores[labels == 1]):.2f}. "
        f"Image AUROC {found['image_auroc']:.3f}, AUPRO {found['aupro']:.3f}. "
        "One colour scale for all panels."
    )
    draw_gallery(panel_rows, title, subtitle, out.figures / "failures.jpg")
    write_json(out.figures / "failures.json", {"panels": records})
    return metrics


def log_evaluation(metrics: dict[str, float], out: Outputs) -> None:
    """Send what the stage wrote to the active MLflow run."""
    mlflow.log_metrics(metrics)
    mlflow.log_artifact(str(out.threshold))
    mlflow.log_artifacts(str(out.figures), "figures")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", required=True)
    args = parser.parse_args(argv)

    params = load_params()
    manifest = pd.read_parquet(MANIFEST_PATH)
    rows = manifest[manifest["category"] == args.category]
    model_file = Path("models/patchcore") / args.category / "model.pt"
    saved = load_patchcore(model_file)
    out = Outputs.for_category(args.category)

    def run() -> dict[str, float]:
        return evaluate_patchcore(
            saved.model,
            saved.image_size,  # the size the model was trained at, from the model file
            rows,
            params.threshold,
            out,
            f"{args.category}: the closest calls",
        )

    if tracking_enabled():
        with start_run("patchcore", f"patchcore-{args.category}", args.category, params):
            # register_model.py finds this run again from the model file's md5.
            mlflow.set_tag("model_md5", md5_of(model_file))
            metrics = run()
            log_evaluation(metrics, out)
    else:
        metrics = run()
    print(f"{args.category}: " + ", ".join(f"{k} {v:.4f}" for k, v in metrics.items()))


if __name__ == "__main__":
    main()
