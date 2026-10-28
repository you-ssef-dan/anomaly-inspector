"""Pictures of the metrics, saved with each MLflow run so a number can be explained later.

A Figure built directly (not through pyplot) belongs to nobody else: no window opens, and it
is freed like any Python object once it is saved.
"""

import numpy as np
from matplotlib.figure import Figure
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import roc_curve

from anomaly_inspector.evaluation.metrics import aupro, image_auroc, pro_curve

# The project's palette (the same as the Phase 2 charts), light theme.
SURFACE, TEXT, MUTED, GRID, LINE = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9", "#2a78d6"


def curve_figure(x: ArrayLike, y: ArrayLike, title: str, xlabel: str, ylabel: str) -> Figure:
    # One curve on a square 0..1 chart, with the diagonal that a random model would draw.
    fig = Figure(figsize=(5, 5), dpi=110, facecolor=SURFACE)
    ax = fig.subplots()
    ax.set_facecolor(SURFACE)
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle="--")  # chance
    ax.plot(x, y, color=LINE, linewidth=2)
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel=xlabel, ylabel=ylabel)
    ax.set_title(title, color=TEXT, loc="left", fontsize=11)
    ax.grid(color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    return fig


Curve = tuple[NDArray[np.float64], NDArray[np.float64]]  # (x, y)


def roc_points(labels: ArrayLike, scores: ArrayLike) -> Curve:
    # Every possible threshold gives one point: how many good parts it rejects (x)
    # against how many defective parts it catches (y). Image AUROC is the area under it.
    fpr, tpr, _ = roc_curve(labels, scores)
    return fpr.astype(np.float64), tpr.astype(np.float64)


def pro_points(masks: ArrayLike, maps: ArrayLike) -> Curve:
    # Every threshold on the anomaly map gives one point: the share of normal pixels it
    # flags (x) against how much of each defect region it covers, on average (y).
    # The curve has one point per pixel value (millions); 1001 points draw the same line.
    # It starts at (0, 0) and first climbs straight up, so that point is kept as it is.
    fpr, pro = pro_curve(masks, maps)
    grid = np.linspace(0, 1, 1001)
    return np.concatenate([[0.0], grid]), np.concatenate([[0.0], np.interp(grid, fpr, pro)])


def roc_figure(labels: ArrayLike, scores: ArrayLike) -> Figure:
    fpr, tpr = roc_points(labels, scores)
    return curve_figure(
        fpr,
        tpr,
        f"ROC curve, image AUROC {image_auroc(labels, scores):.3f}",
        "good parts rejected (false positive rate)",
        "defective parts caught (true positive rate)",
    )


def pro_figure(masks: ArrayLike, maps: ArrayLike, fpr_limit: float = 0.3) -> Figure:
    # AUPRO is the area under the curve left of fpr_limit, so that part is shaded.
    fpr, pro = pro_points(masks, maps)
    fig = curve_figure(
        fpr,
        pro,
        f"PRO curve, AUPRO {aupro(masks, maps, fpr_limit):.3f}",
        "normal pixels flagged (false positive rate)",
        "share of each defect region found (PRO)",
    )
    fig.axes[0].axvspan(0, fpr_limit, color=LINE, alpha=0.08, linewidth=0)
    return fig
