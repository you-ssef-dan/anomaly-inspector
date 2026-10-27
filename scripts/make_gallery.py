"""Heatmap gallery: the images our PatchCore comes closest to failing on, for one category.

Top row: the defective images with the lowest scores (closest to being missed). Bottom row: the
good images with the highest scores (closest to a false reject). Both are picked by rank, so no
threshold is needed, and from the calib split, where decisions are made; holdout is not looked at.
The model is the experiments' baseline (WideResNet-50, 256 px, 1% coreset, seed 42).

  uv run python scripts/make_gallery.py --category screw
  -> reports/gallery/screw.jpg and an entry in reports/gallery.json

The picture is a JPEG: these are photographs, and carpet's fine texture makes a PNG larger than
the 1 MB the pre-commit hook allows.
"""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from anomaly_inspector.data.dataset import ManifestDataset
from anomaly_inspector.evaluation.evaluate import evaluate, load_masks, predict
from anomaly_inspector.evaluation.gallery import draw_gallery, failure_panels
from anomaly_inspector.models.backbones import build_backbone
from anomaly_inspector.models.patchcore import PatchCore


def run_category(args: argparse.Namespace) -> dict[str, Any]:
    manifest = pd.read_parquet(args.manifest)
    rows = manifest[manifest["category"] == args.category]
    train, calib = rows[rows["split"] == "train"], rows[rows["split"] == "calib"].reset_index()
    size = args.size

    torch.manual_seed(args.seed)
    model = PatchCore(build_backbone(args.backbone, pretrained=not args.no_pretrained))
    model.fit(DataLoader(ManifestDataset(train, size), batch_size=8), args.coreset_ratio, args.seed)
    predictions = predict(model, DataLoader(ManifestDataset(calib, size), batch_size=8))
    masks = load_masks(calib, size)
    labels, scores = calib["label"].to_numpy(), predictions.scores
    metrics = evaluate(labels, predictions, masks)
    panel_rows, records = failure_panels(calib, labels, scores, predictions.maps, masks, args.k)

    good, bad = scores[labels == 0], scores[labels == 1]
    subtitle = (
        f"PatchCore baseline, calib split ({len(calib)} images). Median image score: good "
        f"{np.median(good):.2f}, defective {np.median(bad):.2f}. Image AUROC "
        f"{metrics['image_auroc']:.3f}, AUPRO {metrics['aupro']:.3f}. "
        "One colour scale for all panels."
    )
    draw_gallery(
        panel_rows,
        f"{args.category}: the closest calls",
        subtitle,
        args.out_dir / f"{args.category}.jpg",
    )
    return {
        **metrics,
        "calib_images": len(calib),
        "median_score_good": float(np.median(good)),
        "median_score_defective": float(np.median(bad)),
        "panels": records,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--category", required=True)
    parser.add_argument("--k", type=int, default=4, help="panels per row")
    parser.add_argument("--coreset-ratio", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backbone", default="wide_resnet50_2")
    parser.add_argument("--no-pretrained", action="store_true", help="random weights, for tests")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/manifest.parquet"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports/gallery"))
    parser.add_argument("--json", type=Path, default=Path("reports/gallery.json"))
    args = parser.parse_args(argv)

    result = run_category(args)
    results = json.loads(args.json.read_text()) if args.json.exists() else {}
    results[args.category] = result
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(
        f"{args.category}: image AUROC {result['image_auroc']:.4f} | pixel AUROC "
        f"{result['pixel_auroc']:.4f} | AUPRO {result['aupro']:.4f} -> "
        f"{args.out_dir / (args.category + '.jpg')}",
        flush=True,
    )


if __name__ == "__main__":
    main()
