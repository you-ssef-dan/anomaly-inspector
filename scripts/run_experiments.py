"""Phase 2 experiments with our PatchCore: coreset ratio, number of good images, backbone, size.

Each run changes one setting of the baseline (coreset 0.01, every train image, WideResNet-50,
256 px) and is scored on the calib split: these results choose settings, so holdout stays
untouched for the final numbers. Maps are compared with the masks at 256 x 256 whatever size the
model ran at, so image sizes are comparable. One extra run combines two of the changes.

The "images" runs keep the memory bank the size of the baseline's: with a fixed 1% coreset, one
image would leave only 10 patches, which measures a tiny bank, not few parts. They are repeated
on 3 random subsets, since a single image depends on which image it is.

  uv run python scripts/run_experiments.py --category screw          # every run, resumable
  uv run python scripts/run_experiments.py --category screw --list   # the plan
  uv run python scripts/run_experiments.py --category screw --run images/5/seed1
"""

import argparse
import json
import resource
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader

from anomaly_inspector.data.dataset import ManifestDataset
from anomaly_inspector.evaluation.evaluate import evaluate, load_masks, predict
from anomaly_inspector.models.backbones import build_backbone
from anomaly_inspector.models.patchcore import PatchCore

BASELINE: dict[str, Any] = {
    "coreset_ratio": 0.01,
    "images": "all",
    "subset_seed": None,
    "backbone": "wide_resnet50_2",
    "size": 256,
}
EVAL_SIZE = 256
CORESET_SEED = 42
STRIDE = 8  # layer2 of both backbones is 1/8 of the image: one patch per 8 x 8 pixels


def plan() -> dict[str, dict[str, Any]]:
    """Every run's settings, keyed by name: the baseline plus one change at a time."""
    runs = {"baseline": {}}
    runs |= {f"coreset/{r}": {"coreset_ratio": r} for r in (0.001, 0.1)}
    runs |= {
        f"images/{n}/seed{s}": {"images": n, "subset_seed": s}
        for n in (1, 5, 10, 50)
        for s in (0, 1, 2)
    }
    runs |= {"backbone/resnet18": {"backbone": "resnet18"}}
    runs |= {f"size/{size}": {"size": size} for size in (224, 320)}
    # The one combination: the cheap backbone with the resolution that helped screw most.
    runs |= {"combo/resnet18-320": {"backbone": "resnet18", "size": 320}}
    return {name: BASELINE | change for name, change in runs.items()}


def run(category: str, name: str, manifest_path: Path, batch_size: int) -> dict[str, Any]:
    settings = plan()[name]
    manifest = pd.read_parquet(manifest_path)
    rows = manifest[manifest["category"] == category]
    all_train = rows[rows["split"] == "train"]
    calib = rows[rows["split"] == "calib"]

    ratio = settings["coreset_ratio"]
    train = all_train
    if settings["images"] != "all":
        train = all_train.sample(n=settings["images"], random_state=settings["subset_seed"])
        # Same bank size as the baseline, so only the variety of good parts changes.
        baseline_bank = int(len(all_train) * (EVAL_SIZE // STRIDE) ** 2 * BASELINE["coreset_ratio"])
        ratio = min(1.0, baseline_bank / (len(train) * (settings["size"] // STRIDE) ** 2))

    size = settings["size"]
    torch.manual_seed(CORESET_SEED)
    model = PatchCore(build_backbone(settings["backbone"]))
    start = time.perf_counter()
    model.fit(DataLoader(ManifestDataset(train, size), batch_size=batch_size), ratio, CORESET_SEED)
    fit_seconds = time.perf_counter() - start
    predictions = predict(model, DataLoader(ManifestDataset(calib, size), batch_size=batch_size))
    return {
        **evaluate(calib["label"], predictions, load_masks(calib, EVAL_SIZE)),
        "fit_seconds": fit_seconds,
        "ms_per_image": predictions.ms_per_image,
        "memory_bank_rows": int(model.memory_bank.shape[0]),
        "memory_bank_mb": model.memory_bank.numel() * 4 / 1e6,
        "peak_ram_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "train_images": len(train),
        "calib_images": len(calib),
        "coreset_ratio_used": ratio,
        "settings": settings,
    }


def run_all(category: str, out: Path, extra: Sequence[str]) -> None:
    """Each run in its own process, so its feature bank is freed; finished runs are skipped."""
    for name in plan():
        done = json.loads(out.read_text()) if out.exists() else {}
        if f"{category}/{name}" in done:
            print(f"skip {category}/{name}: already done", flush=True)
            continue
        command = [sys.executable, __file__, "--category", category, "--run", name, *extra]
        subprocess.run(command, check=True)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--category", required=True)
    parser.add_argument("--run", choices=list(plan()), help="one run; without it, every run")
    parser.add_argument("--list", action="store_true", help="print the plan and stop")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/manifest.parquet"))
    parser.add_argument("--out", type=Path, default=Path("reports/experiments.json"))
    args = parser.parse_args(argv)

    if args.list:
        for name, settings in plan().items():
            print(f"{name:20} {settings}")
        return
    if args.run is None:
        extra = ["--batch-size", str(args.batch_size), "--manifest", str(args.manifest)]
        run_all(args.category, args.out, [*extra, "--out", str(args.out)])
        return

    result = run(args.category, args.run, args.manifest, args.batch_size)
    results = json.loads(args.out.read_text()) if args.out.exists() else {}
    results[f"{args.category}/{args.run}"] = result
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(
        f"{args.category}/{args.run}: image AUROC {result['image_auroc']:.4f}"
        f" | pixel AUROC {result['pixel_auroc']:.4f} | AUPRO {result['aupro']:.4f}"
        f" | bank {result['memory_bank_rows']} | fit {result['fit_seconds']:.0f} s"
        f" | {result['ms_per_image']:.0f} ms/image | peak RAM {result['peak_ram_mb']:.0f} MB",
        flush=True,
    )


if __name__ == "__main__":
    main()
