"""Heatmap gallery: the images a model comes closest to failing on, as one picture.

Top row: the defective images with the lowest scores (closest to being missed). Bottom row: the
good images with the highest scores (closest to a false reject). Both are picked by rank, so no
threshold is needed. Used by scripts/make_gallery.py (calib) and by each MLflow run (holdout).

The picture is a JPEG: these are photographs, and carpet's fine texture makes a PNG larger than
the 1 MB the pre-commit hook allows.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from numpy.typing import ArrayLike, NDArray
from PIL import Image

from anomaly_inspector.evaluation.failures import (
    defect_percent,
    highest_scoring_goods,
    lowest_scoring_defects,
    rank_among,
)

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
    fig = Figure(
        figsize=(2.25 * columns + 0.7, 2.75 * len(rows) + 1.25), dpi=DPI, facecolor=SURFACE
    )
    axes = fig.subplots(len(rows), columns, squeeze=False)
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
    # matplotlib draws a tuple of lines as one swatch, but its type hints only know Artists
    handles: list[Any] = [outline, Line2D([], [], color=HEAT, linewidth=6, alpha=0.8)]
    fig.legend(
        handles,
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
    return path


def load_image(path: str, size: int) -> NDArray[np.uint8]:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR))


def failure_panels(
    rows: pd.DataFrame,
    labels: ArrayLike,
    scores: NDArray[np.float32],
    maps: NDArray[np.float32],
    masks: NDArray[np.bool_],
    k: int,
) -> tuple[list[list[Panel]], list[dict[str, Any]]]:
    """The two rows of panels (closest misses, closest false rejects) and the facts behind each.

    rows, labels, scores, maps and masks describe the same images in the same order. All panels
    share one colour scale, from a typical good part (0) to the strongest map shown (1).
    """
    rows, labels_a = rows.reset_index(drop=True), np.asarray(labels)
    size = masks.shape[-1]
    picks = {
        "missed": lowest_scoring_defects(labels_a, scores, k),
        "false_reject": highest_scoring_goods(labels_a, scores, k),
    }
    low = float(np.median(maps[labels_a == 0]))  # the level of a typical good part
    high = float(max(maps[i].max() for group in picks.values() for i in group))

    panels: dict[str, list[Panel]] = {}
    records: list[dict[str, Any]] = []
    for kind, positions in picks.items():
        panels[kind] = []
        for i in positions:
            row, defect = rows.iloc[i], kind == "missed"
            peak = np.unravel_index(int(maps[i].argmax()), maps[i].shape)
            rank = rank_among(labels_a, scores, int(i), highest=not defect)
            n_class = int((labels_a == (1 if defect else 0)).sum())
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
    return [panels["missed"], panels["false_reject"]], records
