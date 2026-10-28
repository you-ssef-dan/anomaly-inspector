"""DVC stage train_patchcore@<category>: fit PatchCore on the category's good images, save it.

  python -m anomaly_inspector.pipelines.train_patchcore --category bottle
  -> models/patchcore/bottle/model.pt

Only the train split is read. Scoring, the threshold and the reports are the evaluate stage's
job, so changing a threshold setting does not refit the model.
"""

import argparse
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from anomaly_inspector.config import Params, load_params
from anomaly_inspector.data.dataset import ManifestDataset
from anomaly_inspector.models.backbones import build_backbone
from anomaly_inspector.models.patchcore import PatchCore
from anomaly_inspector.models.storage import save_patchcore

MANIFEST_PATH = Path("data/processed/manifest.parquet")
MODELS_DIR = Path("models/patchcore")


def model_path(category: str) -> Path:
    return MODELS_DIR / category / "model.pt"


def train_patchcore(params: Params, train: pd.DataFrame) -> PatchCore:
    """A PatchCore whose memory bank holds a coreset of the train images' patch features."""
    torch.manual_seed(params.patchcore.seed)
    model = PatchCore(build_backbone(params.patchcore.backbone))
    images = DataLoader(ManifestDataset(train, params.data.image_size), batch_size=8)
    model.fit(images, params.patchcore.coreset_ratio, params.patchcore.seed)
    return model


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", required=True)
    args = parser.parse_args(argv)

    params = load_params()
    manifest = pd.read_parquet(MANIFEST_PATH)
    train = manifest[(manifest["category"] == args.category) & (manifest["split"] == "train")]
    if train.empty:
        raise ValueError(f"no train images for category {args.category!r} in {MANIFEST_PATH}")

    model = train_patchcore(params, train)
    path = save_patchcore(
        model, params.patchcore.backbone, params.data.image_size, model_path(args.category)
    )
    print(f"{args.category}: memory bank {tuple(model.memory_bank.shape)} -> {path}")


if __name__ == "__main__":
    main()
