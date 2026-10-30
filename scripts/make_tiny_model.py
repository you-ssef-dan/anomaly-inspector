"""Write a tiny random PatchCore ONNX model, so CI can test the API image without the real one.

    uv run python scripts/make_tiny_model.py models/patchcore/bottle/model.onnx

The real model lives in DVC, which CI cannot read yet. This one is a ResNet-18 with random
weights fitted on 4 random 64 px images: it has no quality, only the same shape and metadata as
the real one (the same code as the API tests). It checks the image, never the model.
"""

import sys
from pathlib import Path

import torch

from anomaly_inspector.export.onnx_export import export_onnx, write_metadata
from anomaly_inspector.models.backbones import build_backbone
from anomaly_inspector.models.patchcore import PatchCore

SIZE = 64


def main(out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    model = PatchCore(build_backbone("resnet18", pretrained=False))
    images = torch.rand(4, 3, SIZE, SIZE, generator=torch.Generator().manual_seed(1))
    model.fit([(images, torch.zeros(4))], coreset_ratio=0.5, seed=0)
    path = export_onnx(model, out, SIZE)
    write_metadata(
        path,
        {"threshold": "1.5", "image_size": str(SIZE), "category": "bottle", "git_sha": "tiny"},
    )
    print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
