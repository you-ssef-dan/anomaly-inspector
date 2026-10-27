"""Charts and a summary table for reports/experiments.json (written by run_experiments.py).

  uv run python scripts/plot_experiments.py

Writes reports/figures/<experiment>_{light,dark}.png and reports/experiments.md. In a README,
show the right one per theme with <picture> and prefers-color-scheme.
"""

import json
from collections.abc import Sequence
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

RESULTS = Path("reports/experiments.json")
FIGURES = Path("reports/figures")
TABLE = Path("reports/experiments.md")
DPI = 150
PX = 72 / DPI  # one pixel, in points

# Validated palette (dataviz reference instance). A category keeps its colour on every chart.
THEMES: dict[str, dict[str, Any]] = {
    "light": {
        "surface": "#fcfcfb",
        "text": "#0b0b0b",
        "text2": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "series": {"bottle": "#2a78d6", "screw": "#eb6834", "carpet": "#1baf7a"},
    },
    "dark": {
        "surface": "#1a1a19",
        "text": "#f0efec",
        "text2": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "series": {"bottle": "#3987e5", "screw": "#d95926", "carpet": "#199e70"},
    },
}
CATEGORY_ORDER = ["bottle", "screw", "carpet"]
METRICS = {"aupro": "AUPRO", "image_auroc": "Image AUROC"}


def runs_of(results: dict[str, Any], category: str) -> dict[str, Any]:
    prefix = f"{category}/"
    return {k.removeprefix(prefix): v for k, v in results.items() if k.startswith(prefix)}


def series(
    results: dict[str, Any], category: str, experiment: str
) -> list[tuple[str, list[dict[str, Any]]]]:
    """(x label, the runs at that x) in display order; the baseline joins every experiment."""
    runs = runs_of(results, category)
    base = runs.get("baseline")
    points: dict[str, list[dict[str, Any]]] = {}
    if experiment == "coreset":
        for ratio, key in ((0.001, "coreset/0.001"), (0.01, "baseline"), (0.1, "coreset/0.1")):
            points[f"{ratio:.1%}".replace(".0%", "%")] = [runs[key]] if key in runs else []
    elif experiment == "images":
        for n in (1, 5, 10, 50):
            points[str(n)] = [v for k, v in runs.items() if k.startswith(f"images/{n}/")]
        points["all"] = [base] if base else []
    elif experiment == "backbone":
        points["ResNet-18"] = [runs["backbone/resnet18"]] if "backbone/resnet18" in runs else []
        points["WideResNet-50"] = [base] if base else []
    elif experiment == "size":
        for size, key in ((224, "size/224"), (256, "baseline"), (320, "size/320")):
            points[f"{size} px"] = [runs[key]] if key in runs else []
    return [(x, rs) for x, rs in points.items() if rs]


def style(ax: Axes, theme: dict[str, Any]) -> None:
    ax.set_facecolor(theme["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme["axis"])
        ax.spines[side].set_linewidth(1 * PX)
    ax.tick_params(colors=theme["muted"], labelcolor=theme["text2"], labelsize=8, width=1 * PX)
    ax.grid(axis="y", color=theme["grid"], linewidth=1 * PX, linestyle="-")
    ax.set_axisbelow(True)


def draw(
    ax: Axes, results: dict[str, Any], experiment: str, key: str, theme: dict[str, Any], label: str
) -> None:
    style(ax, theme)
    lowest = float("inf")
    for category in CATEGORY_ORDER:
        points = series(results, category, experiment)
        if not points:
            continue
        color = theme["series"][category]
        xs = range(len(points))
        values = [[run[key] for run in runs] for _, runs in points]
        middle = [mean(v) for v in values]
        lowest = min(lowest, *(min(v) for v in values))
        if any(len(v) > 1 for v in values):  # several random subsets: show their spread
            ax.fill_between(
                xs,
                [min(v) for v in values],
                [max(v) for v in values],
                color=color,
                alpha=0.10,
                linewidth=0,
            )
        ax.plot(
            xs,
            middle,
            color=color,
            linewidth=2 * PX,
            marker="o",
            markersize=10 * PX,
            markeredgecolor=theme["surface"],
            markeredgewidth=2 * PX,
            solid_capstyle="round",
            solid_joinstyle="round",
        )
        ax.annotate(
            category,
            (xs[-1], middle[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color=theme["text2"],
        )
        ax.set_xticks(list(xs), [x for x, _ in points])
    ax.set_title(label, loc="left", fontsize=9, color=theme["text"], pad=6)
    ax.margins(x=0.12)
    if key in METRICS:
        # Scores stop at 1. Show at least 0.10 of range, so a 0.005 gap does not fill the panel.
        top = 1.005
        ax.set_ylim(min(lowest - 0.01, top - 0.10), top)
    else:
        ax.set_ylim(bottom=0)  # a cost: compare magnitudes from zero


TITLES = {
    "coreset": ("Coreset ratio", "Share of patch features kept in the memory bank"),
    "images": (
        "Number of good training images",
        "Memory bank kept at the baseline's size; band = 3 random subsets (min to max)",
    ),
    "backbone": ("Backbone", "Pretrained ImageNet network that turns patches into features"),
    "size": ("Image size", "Model input size; every map is scored on 256 px masks"),
}
COST = {
    "coreset": ("ms_per_image", "ms per image"),
    "backbone": ("ms_per_image", "ms per image"),
    "size": ("ms_per_image", "ms per image"),
    "images": None,
}


def plot(results: dict[str, Any], experiment: str, theme_name: str) -> Path:
    theme = THEMES[theme_name]
    panels: list[tuple[str, str]] = list(METRICS.items())
    if COST[experiment]:
        panels.append(COST[experiment])
    fig, axes = plt.subplots(1, len(panels), figsize=(3.4 * len(panels), 3.1), dpi=DPI)
    fig.patch.set_facecolor(theme["surface"])
    for ax, (key, label) in zip(axes, panels, strict=True):
        draw(ax, results, experiment, key, theme, label)
    title, subtitle = TITLES[experiment]
    fig.text(0.01, 0.97, title, fontsize=11, fontweight="bold", color=theme["text"], va="top")
    fig.text(
        0.01,
        0.905,
        f"{subtitle}. PatchCore, scored on the calib split.",
        fontsize=8,
        color=theme["text2"],
        va="top",
    )
    shown = [c for c in CATEGORY_ORDER if series(results, c, experiment)]
    handles = [
        Line2D(
            [],
            [],
            color=theme["series"][c],
            linewidth=2 * PX,
            marker="o",
            markersize=8 * PX,
            markeredgecolor=theme["surface"],
        )
        for c in shown
    ]
    legend = fig.legend(
        handles,
        shown,
        loc="upper right",
        ncols=len(shown),
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.99, 0.99),
    )
    for text in legend.get_texts():
        text.set_color(theme["text2"])
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / f"experiment_{experiment}_{theme_name}.png"
    fig.savefig(path, facecolor=theme["surface"])
    plt.close(fig)
    return path


def table(results: dict[str, Any]) -> str:
    lines = [
        "# Phase 2 experiments (PatchCore, calib split)",
        "",
        "Generated by `scripts/plot_experiments.py` from `reports/experiments.json`."
        " Image runs show the mean of 3 random subsets.",
        "",
        "| Category | Run | Image AUROC | Pixel AUROC | AUPRO | Bank (MB) | Fit (s) | ms/image |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for category in CATEGORY_ORDER:
        runs = runs_of(results, category)
        groups: dict[str, list[dict[str, Any]]] = {}
        for name, run in runs.items():
            groups.setdefault(name.rsplit("/seed", 1)[0], []).append(run)
        for name in sorted(groups, key=lambda n: (n != "baseline", n)):
            rs = groups[name]

            def avg(key: str, runs: Sequence[dict[str, Any]] = rs) -> float:
                return mean(r[key] for r in runs)

            lines.append(
                f"| {category} | {name} | {avg('image_auroc'):.4f} | {avg('pixel_auroc'):.4f} "
                f"| {avg('aupro'):.4f} | {avg('memory_bank_mb'):.1f} | {avg('fit_seconds'):.0f} "
                f"| {avg('ms_per_image'):.0f} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    results = json.loads(RESULTS.read_text())
    for experiment in TITLES:
        for theme_name in THEMES:
            print("wrote", plot(results, experiment, theme_name))
    TABLE.write_text(table(results))
    print("wrote", TABLE)


if __name__ == "__main__":
    main()
