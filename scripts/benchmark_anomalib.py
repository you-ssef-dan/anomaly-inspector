"""Compare our PatchCore with anomalib's on identical inputs, scored by our own metrics.

Both get the same preprocessed tensors, the same training images (split "train"), the same
test images (calib + holdout), the same coreset ratio and the same metric code, so a difference
in the numbers comes from the model code alone.

anomalib needs the optional `bench` extra: `uv sync --all-extras`. Implementations:

  ours             this repo's PatchCore
  anomalib-v1      anomalib's PatchCore with the torchvision V1 weights loaded into its backbone
                   (the weights of the paper and of ours: the algorithm is the only difference)
  anomalib-default anomalib's PatchCore as shipped; its timm backbone uses `racm_in1k` weights

Run one (category, implementation) per process so the memory of the feature bank is released:

  uv run python scripts/benchmark_anomalib.py --category bottle --impl ours
  uv run python scripts/benchmark_anomalib.py --category bottle --impl anomalib-v1
"""

import argparse
import json
import resource
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader

from anomaly_inspector.data.dataset import ManifestDataset
from anomaly_inspector.evaluation.evaluate import Predictions, evaluate, load_masks, predict
from anomaly_inspector.models.backbones import build_backbone
from anomaly_inspector.models.patchcore import PatchCore

IMPLS = ("ours", "anomalib-v1", "anomalib-default")
LAYERS = ("layer2", "layer3")
ANOMALIB_DEFAULT_NEIGHBORS = 9  # anomalib re-weights the image score over this many neighbours


def run_ours(
    backbone: torch.nn.Module,
    train: DataLoader[Any],
    test: DataLoader[Any],
    ratio: float,
    seed: int,
) -> tuple[Predictions, float]:
    torch.manual_seed(seed)
    model = PatchCore(backbone, layers=LAYERS)
    start = time.perf_counter()
    model.fit(train, ratio, seed=seed)
    fit_seconds = time.perf_counter() - start
    return predict(model, test), fit_seconds


class AnomalibAsTuple:
    """anomalib returns an InferenceBatch; present it like ours: (anomaly_map, image_score)."""

    def __init__(self, model: Any) -> None:
        self.model = model

    @torch.no_grad()
    def __call__(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.model(images)
        return out.anomaly_map, out.pred_score


def run_anomalib(
    impl: str,
    backbone_name: str,
    source_backbone: torch.nn.Module,
    train: DataLoader[Any],
    test: DataLoader[Any],
    ratio: float,
    seed: int,
    num_neighbors: int,
    pretrained: bool,
) -> tuple[Predictions, float]:
    from anomalib.models.image.patchcore.torch_model import PatchcoreModel

    torch.manual_seed(seed)
    own_weights = impl == "anomalib-default" and pretrained  # its timm weights, downloaded
    model = PatchcoreModel(
        layers=LAYERS, backbone=backbone_name, pre_trained=own_weights, num_neighbors=num_neighbors
    )
    if impl == "anomalib-v1":
        load_weights_and_check(model, source_backbone)

    start = time.perf_counter()
    model.train()  # collects the patch embeddings; its extractor stays in eval mode by itself
    with torch.no_grad():
        for images, _ in train:
            model(images)
    torch.manual_seed(seed)  # the coreset's random start, whatever the weights consumed before
    model.subsample_embedding(ratio)
    fit_seconds = time.perf_counter() - start
    model.eval()
    return predict(AnomalibAsTuple(model), test), fit_seconds


def load_weights_and_check(anomalib_model: Any, source: torch.nn.Module) -> None:
    """Put our backbone's weights into anomalib's timm backbone and prove the features match."""
    extractor = anomalib_model.feature_extractor
    missing, _ = extractor.feature_extractor.load_state_dict(source.state_dict(), strict=False)
    if missing:
        raise RuntimeError(f"anomalib's backbone is missing weights: {missing[:3]}")
    probe = torch.randn(2, 3, 128, 128, generator=torch.Generator().manual_seed(0))
    ours = PatchCore(source, layers=LAYERS).extractor
    with torch.no_grad():
        theirs, reference = extractor.eval()(probe), ours(probe)
    for layer in LAYERS:
        if not torch.equal(theirs[layer], reference[layer]):
            raise RuntimeError(f"features of {layer} differ after loading the weights")
    print("anomalib backbone loaded with our weights: features are identical", flush=True)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--category", required=True)
    parser.add_argument("--impl", choices=IMPLS, required=True)
    parser.add_argument("--coreset-ratio", type=float, default=0.01)
    parser.add_argument(
        "--num-neighbors",
        type=int,
        default=ANOMALIB_DEFAULT_NEIGHBORS,
        help="anomalib only; 1 gives the plain maximum patch distance, like ours",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backbone", default="wide_resnet50_2")
    parser.add_argument(
        "--no-pretrained", action="store_true", help="random weights, for smoke tests"
    )
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/manifest.parquet"))
    parser.add_argument("--out", type=Path, default=Path("reports/anomalib_benchmark.json"))
    args = parser.parse_args(argv)

    manifest = pd.read_parquet(args.manifest)
    rows = manifest[manifest["category"] == args.category]
    train_rows = rows[rows["split"] == "train"]
    test_rows = rows[rows["split"].isin(["calib", "holdout"])]  # the full test set
    train = DataLoader(ManifestDataset(train_rows, args.size), batch_size=args.batch_size)
    test = DataLoader(ManifestDataset(test_rows, args.size), batch_size=args.batch_size)
    pretrained = not args.no_pretrained
    source = build_backbone(args.backbone, pretrained=pretrained)

    if args.impl == "ours":
        predictions, fit_seconds = run_ours(source, train, test, args.coreset_ratio, args.seed)
    else:
        predictions, fit_seconds = run_anomalib(
            args.impl,
            args.backbone,
            source,
            train,
            test,
            args.coreset_ratio,
            args.seed,
            args.num_neighbors,
            pretrained,
        )

    masks = load_masks(test_rows, args.size)
    result = {
        "category": args.category,
        "impl": args.impl,
        **evaluate(test_rows["label"], predictions, masks),
        "fit_seconds": fit_seconds,
        "ms_per_image": predictions.ms_per_image,  # time in the model only, not image decoding
        "peak_ram_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "train_images": len(train_rows),
        "test_images": len(test_rows),
        "settings": {
            "coreset_ratio": args.coreset_ratio,
            "seed": args.seed,
            "backbone": args.backbone,
            "pretrained": pretrained,
            "image_size": args.size,
            "torch": torch.__version__,
            "num_neighbors": args.num_neighbors if args.impl != "ours" else None,
        },
    }
    key = f"{args.category}/{args.impl}"
    if args.impl != "ours":
        key += f"-k{args.num_neighbors}"
    results = json.loads(args.out.read_text()) if args.out.exists() else {}
    results[key] = result
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(
        f"{key}: image AUROC {result['image_auroc']:.4f}"
        f" | pixel AUROC {result['pixel_auroc']:.4f}"
        f" | AUPRO {result['aupro']:.4f}"
        f" | fit {fit_seconds:.0f} s"
        f" | {result['ms_per_image']:.0f} ms/image"
        f" | peak RAM {result['peak_ram_mb']:.0f} MB"
    )


if __name__ == "__main__":
    main()
