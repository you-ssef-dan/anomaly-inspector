"""Compare our PatchCore and our STFPM with anomalib's PatchCore, on identical inputs and metrics.

All get the same preprocessed tensors, the same training images (split "train"), the same test
images (calib + holdout) and the same metric code, so a difference in the numbers comes from the
model alone. The two PatchCores also share the coreset ratio and the backbone's weights.

anomalib needs the optional `bench` extra: `uv sync --all-extras`. Implementations:

  ours             this repo's PatchCore (WideResNet-50)
  anomalib-v1      anomalib's PatchCore with the torchvision V1 weights loaded into its backbone
                   (the weights of the paper and of ours: the algorithm is the only difference)
  anomalib-default anomalib's PatchCore as shipped; its timm backbone uses `racm_in1k` weights
  stfpm            this repo's STFPM (ResNet-18), trained with the stfpm section of params.yaml:
                   early stopping on split "val_normal", seeded, so a rerun gives the same numbers

Run one (category, implementation) per process so the memory of the feature bank is released:

  uv run python scripts/benchmark_anomalib.py --category bottle --impl ours
  uv run python scripts/benchmark_anomalib.py --category bottle --impl anomalib-v1
  uv run --env-file .env python scripts/benchmark_anomalib.py --category bottle --impl stfpm

With a tracking server (`--env-file .env`), an STFPM run is logged to MLflow, experiment "stfpm":
its settings, the losses of every epoch, and the scores.
"""

import argparse
import contextlib
import json
import resource
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
import torch
from torch.utils.data import DataLoader

from anomaly_inspector.config import StfpmConfig, load_params
from anomaly_inspector.data.dataset import ManifestDataset
from anomaly_inspector.evaluation.evaluate import Predictions, evaluate, load_masks, predict
from anomaly_inspector.models.backbones import build_backbone
from anomaly_inspector.models.patchcore import PatchCore
from anomaly_inspector.models.stfpm import STFPM
from anomaly_inspector.tracking import start_run, tracking_enabled
from anomaly_inspector.training.stfpm import TrainResult, stack_images, train_stfpm

IMPLS = ("ours", "anomalib-v1", "anomalib-default", "stfpm")
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


def run_stfpm(
    backbone_name: str,
    pretrained: bool,
    train: ManifestDataset,
    val: ManifestDataset,
    test: DataLoader[Any],
    config: StfpmConfig,
) -> tuple[Predictions, float, TrainResult]:
    torch.manual_seed(config.seed)  # the student's random starting weights
    model = STFPM.from_backbone(backbone_name, pretrained=pretrained)
    start = time.perf_counter()
    train_images, val_images = stack_images(train), stack_images(val)

    def log_epoch(epoch: int, metrics: dict[str, float]) -> None:
        print(f"epoch {epoch}: " + ", ".join(f"{k} {v:.4f}" for k, v in metrics.items()))
        if tracking_enabled():
            losses = {"train_loss": metrics["train_loss"], "val_loss": metrics["val_loss"]}
            mlflow.log_metrics(losses | {"epoch_seconds": metrics["seconds"]}, step=epoch)

    with tempfile.TemporaryDirectory() as folder:  # the best student only lives until it is scored
        trained = train_stfpm(
            model, train_images, val_images, config, Path(folder) / "student.pt", log_epoch
        )
    fit_seconds = time.perf_counter() - start
    return predict(model.predict, test), fit_seconds, trained


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
    parser.add_argument("--coreset-ratio", type=float, default=0.01, help="PatchCore only")
    parser.add_argument(
        "--num-neighbors",
        type=int,
        default=ANOMALIB_DEFAULT_NEIGHBORS,
        help="anomalib only; 1 gives the plain maximum patch distance, like ours",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--backbone", help="default: wide_resnet50_2 for PatchCore, resnet18 for STFPM (papers)"
    )
    parser.add_argument(
        "--no-pretrained", action="store_true", help="random weights, for smoke tests"
    )
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8, help="for extracting and scoring")
    parser.add_argument("--epochs", type=int, help="STFPM only; default: the one in params.yaml")
    parser.add_argument("--params", type=Path, default=Path("params.yaml"), help="STFPM only")
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/manifest.parquet"))
    parser.add_argument("--out", type=Path, default=Path("reports/anomalib_benchmark.json"))
    args = parser.parse_args(argv)

    stfpm = args.impl == "stfpm"
    backbone = args.backbone or ("resnet18" if stfpm else "wide_resnet50_2")
    manifest = pd.read_parquet(args.manifest)
    rows = manifest[manifest["category"] == args.category]
    train_rows = rows[rows["split"] == "train"]
    test_rows = rows[rows["split"].isin(["calib", "holdout"])]  # the full test set
    test = DataLoader(ManifestDataset(test_rows, args.size), batch_size=args.batch_size)
    pretrained = not args.no_pretrained
    settings: dict[str, Any] = {
        "seed": args.seed,
        "backbone": backbone,
        "pretrained": pretrained,
        "image_size": args.size,
        "torch": torch.__version__,
    }
    extra: dict[str, Any] = {}
    with contextlib.ExitStack() as tracking:
        if stfpm:
            params = load_params(args.params)
            changes: dict[str, Any] = {"seed": args.seed}
            if args.epochs is not None:
                changes["epochs"] = args.epochs
            params = params.model_copy(update={"stfpm": params.stfpm.model_copy(update=changes)})
            val_rows = rows[rows["split"] == "val_normal"]
            if tracking_enabled():  # open until the scores are logged; FAILED if anything raises
                run = tracking.enter_context(
                    start_run("stfpm", f"stfpm-{args.category}", args.category, params)
                )
            predictions, fit_seconds, trained = run_stfpm(
                backbone,
                pretrained,
                ManifestDataset(train_rows, args.size),
                ManifestDataset(val_rows, args.size),
                test,
                params.stfpm,
            )
            settings["stfpm"] = params.stfpm.model_dump()
            extra = {
                "val_images": len(val_rows),
                "best_epoch": trained.best_epoch,
                "epochs_run": trained.epochs_run,
                "best_val_loss": trained.best_val_loss,
            }
        else:
            source = build_backbone(backbone, pretrained=pretrained)
            train = DataLoader(ManifestDataset(train_rows, args.size), batch_size=args.batch_size)
            if args.impl == "ours":
                predictions, fit_seconds = run_ours(
                    source, train, test, args.coreset_ratio, args.seed
                )
            else:
                predictions, fit_seconds = run_anomalib(
                    args.impl,
                    backbone,
                    source,
                    train,
                    test,
                    args.coreset_ratio,
                    args.seed,
                    args.num_neighbors,
                    pretrained,
                )
            settings["coreset_ratio"] = args.coreset_ratio
            settings["num_neighbors"] = args.num_neighbors if args.impl != "ours" else None

        masks = load_masks(test_rows, args.size)
        scores = evaluate(test_rows["label"], predictions, masks)
        result = {
            "category": args.category,
            "impl": args.impl,
            **scores,
            "fit_seconds": fit_seconds,
            "ms_per_image": predictions.ms_per_image,  # time in the model only, not image decoding
            "peak_ram_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "train_images": len(train_rows),
            "test_images": len(test_rows),
            **extra,
            "settings": settings,
        }
        if stfpm and tracking_enabled():
            mlflow.set_tag("evaluated_on", "calib + holdout")
            mlflow.log_metrics({f"test_{name}": value for name, value in scores.items()})
            mlflow.log_metrics(
                {
                    "fit_seconds": fit_seconds,
                    "ms_per_image": predictions.ms_per_image,
                    "best_epoch": trained.best_epoch,
                    "epochs_run": trained.epochs_run,
                    "best_val_loss": trained.best_val_loss,
                }
            )
            print(f"MLflow run {run.info.run_id}, experiment stfpm")

    key = f"{args.category}/{args.impl}"
    if args.impl.startswith("anomalib"):
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
