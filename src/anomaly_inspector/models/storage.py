"""Save a fitted PatchCore to one file, and load it back without downloading anything.

The file carries the backbone's name and the input size next to the weights, so whoever loads it
(the evaluate stage today, the ONNX export and the API later) never guesses them from params.yaml,
which may have changed since the model was trained.
"""

from dataclasses import dataclass
from pathlib import Path

import torch

from anomaly_inspector.models.backbones import build_backbone
from anomaly_inspector.models.patchcore import PatchCore


@dataclass(frozen=True)
class SavedPatchCore:
    model: PatchCore
    backbone: str
    image_size: int


def save_patchcore(model: PatchCore, backbone: str, image_size: int, path: Path) -> Path:
    # The same model always gives the same bytes, so DVC sees "unchanged" when a re-run
    # trains the same thing, and skips the stages after it.
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"backbone": backbone, "image_size": image_size, "state_dict": model.state_dict()}, path
    )
    return path


def load_patchcore(path: Path) -> SavedPatchCore:
    # weights_only=True: the file may hold only tensors and plain values, so loading it can
    # never run code hidden in it (plain pickle could).
    saved = torch.load(path, weights_only=True)
    # Random weights first: the real ones, backbone included, come from the file.
    model = PatchCore(build_backbone(saved["backbone"], pretrained=False))
    model.load_state_dict(saved["state_dict"])
    return SavedPatchCore(model.eval(), saved["backbone"], int(saved["image_size"]))
