"""DVC stage optimize_patchcore@<category>: an INT8 model, and one speed table for all three.

  python -m anomaly_inspector.pipelines.optimize_patchcore --category bottle

Writes models/patchcore/<category>/model.int8.onnx, calibrated on good train images, and
reports/patchcore/<category>/benchmark.json. For PyTorch FP32, ONNX FP32 and ONNX INT8 it holds
the latency at batch 1 (p50, p95), the images per second at batch 8, the file size, and the
holdout image AUROC and AUPRO, all measured on the same number of CPU threads.

The INT8 model is measured, not shipped: its scores shift a little, so serving it would need its
own threshold. model.onnx (FP32) stays the model that serves.
"""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader

from anomaly_inspector.config import load_params
from anomaly_inspector.data.dataset import ManifestDataset
from anomaly_inspector.evaluation.evaluate import evaluate, load_masks, predict
from anomaly_inspector.export.benchmark import Run, measure
from anomaly_inspector.export.onnx_export import OnnxPatchCore
from anomaly_inspector.export.quantize import quantize_int8
from anomaly_inspector.models.storage import load_patchcore

MANIFEST_PATH = Path("data/processed/manifest.parquet")


def int8_path(category: str) -> Path:
    return Path("models/patchcore") / category / "model.int8.onnx"


def report_path(category: str) -> Path:
    return Path("reports/patchcore") / category / "benchmark.json"


def calibration_images(train: pd.DataFrame, count: int, size: int) -> list[NDArray[np.float32]]:
    """The first train images, preprocessed exactly as for scoring."""
    dataset = ManifestDataset(train.head(count), size)
    return [dataset[i][0].numpy() for i in range(len(dataset))]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", required=True)
    args = parser.parse_args(argv)
    cfg = load_params().export
    models = Path("models/patchcore") / args.category
    saved = load_patchcore(models / "model.pt")
    size = saved.image_size

    manifest = pd.read_parquet(MANIFEST_PATH)
    rows = manifest[manifest["category"] == args.category]
    train = rows[rows["split"] == "train"]
    int8 = quantize_int8(
        models / "model.onnx",
        int8_path(args.category),
        calibration_images(train, cfg.calibration_images, size),
    )

    # INT8 scored on holdout like the others; FP32 numbers come from the export stage.
    holdout = rows[rows["split"] == "holdout"]
    loader = DataLoader(ManifestDataset(holdout, size), batch_size=8)
    found = evaluate(
        holdout["label"].to_numpy(), predict(OnnxPatchCore(int8), loader), load_masks(holdout, size)
    )
    exported = json.loads((Path("reports/patchcore") / args.category / "export.json").read_text())
    accuracy = {
        "pytorch_fp32": {n: exported[f"pytorch_{n}"] for n in ("image_auroc", "aupro")},
        "onnx_fp32": {n: exported[f"onnx_{n}"] for n in ("image_auroc", "aupro")},
        "onnx_int8": {n: found[n] for n in ("image_auroc", "aupro")},
    }

    # Speed, on the same number of threads for every model.
    torch.set_num_threads(cfg.threads)
    model = saved.model

    def run_pytorch(images: NDArray[np.float32]) -> object:
        with torch.no_grad():
            return model(torch.from_numpy(images))

    def run_onnx(path: Path) -> Run:
        session = OnnxPatchCore(path, threads=cfg.threads)
        return lambda images: session(torch.from_numpy(images))

    variants: dict[str, tuple[Run, Path]] = {
        "pytorch_fp32": (run_pytorch, models / "model.pt"),
        "onnx_fp32": (run_onnx(models / "model.onnx"), models / "model.onnx"),
        "onnx_int8": (run_onnx(int8), int8),
    }
    report: dict[str, object] = {"threads": cfg.threads}
    for name, (run, path) in variants.items():
        one, eight = measure(run, batch=1, size=size), measure(run, batch=8, size=size)
        report[name] = {
            "p50_ms": one["p50_ms"],
            "p95_ms": one["p95_ms"],
            "images_per_s_batch8": eight["images_per_s"],
            "size_mb": path.stat().st_size / 1e6,
            **accuracy[name],
        }
        print(f"{args.category} {name}: {report[name]}", flush=True)
    report_path(args.category).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
