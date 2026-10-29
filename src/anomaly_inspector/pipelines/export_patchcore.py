"""DVC stage export_patchcore@<category>: the saved PatchCore as one ONNX file.

  python -m anomaly_inspector.pipelines.export_patchcore --category bottle

Reads model.pt and threshold.json, and the PyTorch holdout metrics the evaluate stage wrote.
Writes models/patchcore/<category>/model.onnx, with its threshold, image size, category and git
SHA as metadata, and reports/patchcore/<category>/export.json, which puts the ONNX model's
holdout image AUROC and AUPRO next to PyTorch's. The stage fails if they differ by 0.001 or more:
a model that scores differently once exported must not ship.
"""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader

from anomaly_inspector.data.dataset import ManifestDataset
from anomaly_inspector.evaluation.evaluate import evaluate, load_masks, predict
from anomaly_inspector.export.onnx_export import OnnxPatchCore, export_onnx, write_metadata
from anomaly_inspector.models.storage import load_patchcore
from anomaly_inspector.tracking import git_sha

MANIFEST_PATH = Path("data/processed/manifest.parquet")
TOLERANCE = 1e-3  # "equal to 3 decimals"
COMPARED = ("image_auroc", "aupro")


def onnx_path(category: str) -> Path:
    return Path("models/patchcore") / category / "model.onnx"


def report_path(category: str) -> Path:
    return Path("reports/patchcore") / category / "export.json"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", required=True)
    args = parser.parse_args(argv)
    models, reports = (
        Path("models/patchcore") / args.category,
        Path("reports/patchcore") / args.category,
    )

    saved = load_patchcore(models / "model.pt")
    threshold = json.loads((models / "threshold.json").read_text())["threshold"]
    path = export_onnx(saved.model, onnx_path(args.category), saved.image_size)
    write_metadata(
        path,
        {
            "category": args.category,
            "image_size": str(saved.image_size),
            "threshold": str(threshold),
            "git_sha": git_sha(),
        },
    )

    # The ONNX model scored on holdout exactly like the evaluate stage scored the PyTorch one.
    manifest = pd.read_parquet(MANIFEST_PATH)
    holdout = manifest[(manifest["category"] == args.category) & (manifest["split"] == "holdout")]
    loader = DataLoader(ManifestDataset(holdout, saved.image_size), batch_size=8)
    predictions = predict(OnnxPatchCore(path), loader)
    onnx_metrics = evaluate(
        holdout["label"].to_numpy(), predictions, load_masks(holdout, saved.image_size)
    )
    pytorch_metrics = json.loads((reports / "metrics.json").read_text())

    report: dict[str, float] = {"onnx_mb": path.stat().st_size / 1e6}
    report["onnx_ms_per_image"] = predictions.ms_per_image
    for name in COMPARED:
        report[f"pytorch_{name}"] = pytorch_metrics[f"holdout_{name}"]
        report[f"onnx_{name}"] = onnx_metrics[name]
    report_path(args.category).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    gaps = {n: abs(report[f"onnx_{n}"] - report[f"pytorch_{n}"]) for n in COMPARED}
    if max(gaps.values()) >= TOLERANCE:
        raise ValueError(f"the ONNX model scores differently from PyTorch on holdout: {gaps}")
    print(f"{args.category}: {path} ({report['onnx_mb']:.0f} MB), holdout matches PyTorch {gaps}")


if __name__ == "__main__":
    main()
