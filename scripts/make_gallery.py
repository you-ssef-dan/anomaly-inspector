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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from numpy.typing import NDArray
from PIL import Image
from torch.utils.data import DataLoader

from anomaly_inspector.data.dataset import ManifestDataset
from anomaly_inspector.evaluation.evaluate import evaluate, load_masks, predict
from anomaly_inspector.evaluation.failures import (
    defect_percent,
    highest_scoring_goods,
    lowest_scoring_defects,
    rank_among,
)
from anomaly_inspector.models.backbones import build_backbone
from anomaly_inspector.models.patchcore import PatchCore

DPI = 110
PX = 72 / DPI  # one pixel, in points
HEAT = (0xEB / 255, 0x68 / 255, 0x34 / 255)  # the orange of the project's palette
SURFACE, TEXT, TEXT2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
ROW_LABELS = ("Lowest-scoring\ndefects", "Highest-scoring\ngood parts")


@dataclass(frozen=True)
class Panel:
    image: NDArray[np.uint8]  # (H, W, 3)
    heat: NDArray[np.float32]  # (H, W), 0 = a typical good part, 1 = the strongest shown
    mask: NDArray[np.bool_] | None  # the ground-truth defect; None for a good part
    caption: str


def heat_overlay(heat: NDArray[np.float32]) -> NDArray[np.float32]:
    """Orange wash whose opacity follows the map, so normal areas show the image untouched."""
    rgba = np.zeros((*heat.shape, 4), dtype=np.float32)
    rgba[..., :3] = HEAT
    rgba[..., 3] = 0.8 * np.clip(heat, 0, 1)
    return rgba


def draw_gallery(rows: Sequence[Sequence[Panel]], title: str, subtitle: str, path: Path) -> Path:
    columns = max(len(row) for row in rows)
    fig, axes = plt.subplots(
        len(rows),
        columns,
        figsize=(2.25 * columns + 0.7, 2.75 * len(rows) + 1.25),
        dpi=DPI,
        squeeze=False,
    )
    fig.patch.set_facecolor(SURFACE)
    for r, row in enumerate(rows):
        for c in range(columns):
            ax = axes[r, c]
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if c >= len(row):
                ax.set_visible(False)
                continue
            panel = row[c]
            ax.imshow(panel.image)
            ax.imshow(heat_overlay(panel.heat))
            if panel.mask is not None:  # a dark line under a light one, visible on any photo
                for color, width in (("black", 2.6), ("white", 1.3)):
                    ax.contour(
                        panel.mask.astype(float), levels=[0.5], colors=color, linewidths=width * PX
                    )
            ax.set_title(panel.caption, fontsize=7, color=TEXT2, loc="left", pad=4)
        axes[r, 0].set_ylabel(ROW_LABELS[r], fontsize=8, color=TEXT2, labelpad=8)
    fig.text(0.012, 0.985, title, fontsize=11, fontweight="bold", color=TEXT, va="top")
    fig.text(0.012, 0.945, subtitle, fontsize=7.5, color=TEXT2, va="top")
    outline = (  # a white line over a dark one, exactly as the outlines are drawn on the photos
        Line2D([], [], color="black", linewidth=2.6 * PX * 1.6),
        Line2D([], [], color="white", linewidth=1.3 * PX * 1.6),
    )
    fig.legend(
        [outline, Line2D([], [], color=HEAT, linewidth=6, alpha=0.8)],
        ["ground-truth defect (outline)", "model's anomaly map (brighter = more anomalous)"],
        handler_map={tuple: HandlerTuple(ndivide=None)},
        loc="lower left",
        ncols=2,
        frameon=False,
        fontsize=7.5,
        bbox_to_anchor=(0.0, 0.0),
        labelcolor=TEXT2,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.9))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE, pil_kwargs={"quality": 90})
    plt.close(fig)
    return path


def load_image(path: str, size: int) -> NDArray[np.uint8]:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR))


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
    labels, scores, maps = calib["label"].to_numpy(), predictions.scores, predictions.maps
    metrics = evaluate(labels, predictions, masks)

    picks = {
        "missed": lowest_scoring_defects(labels, scores, args.k),
        "false_reject": highest_scoring_goods(labels, scores, args.k),
    }
    low = float(np.median(maps[labels == 0]))  # the level of a typical good part
    high = float(max(maps[i].max() for group in picks.values() for i in group))

    panels, records = {}, []
    for kind, positions in picks.items():
        panels[kind] = []
        for i in positions:
            row, defect = calib.iloc[i], kind == "missed"
            peak = np.unravel_index(int(maps[i].argmax()), maps[i].shape)
            rank = rank_among(labels, scores, int(i), highest=not defect)
            n_class = int((labels == (1 if defect else 0)).sum())
            area = defect_percent(masks[i]) if defect else None
            first = f"{row['defect_type']} · {area:.2f}% of image" if defect else "good part"
            panels[kind].append(
                Panel(
                    image=load_image(row["path"], size),
                    heat=np.clip((maps[i] - low) / (high - low), 0, 1).astype(np.float32),
                    mask=masks[i] if defect else None,
                    caption=f"{first}\nscore {scores[i]:.2f} · {'lowest' if defect else 'highest'}"
                    f" {rank} of {n_class}",
                )
            )
            records.append(
                {
                    "kind": kind,
                    "path": row["path"],
                    "defect_type": row["defect_type"],
                    "score": float(scores[i]),
                    "rank_in_class": rank,
                    "images_in_class": n_class,
                    "defect_percent": area,
                    "peak_yx": [int(peak[0]), int(peak[1])],
                    "peak_on_defect": bool(masks[i][peak]) if defect else None,
                }
            )
    good, bad = scores[labels == 0], scores[labels == 1]
    subtitle = (
        f"PatchCore baseline, calib split ({len(calib)} images). Median image score: good "
        f"{np.median(good):.2f}, defective {np.median(bad):.2f}. Image AUROC "
        f"{metrics['image_auroc']:.3f}, AUPRO {metrics['aupro']:.3f}. "
        "One colour scale for all panels."
    )
    draw_gallery(
        [panels["missed"], panels["false_reject"]],
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
