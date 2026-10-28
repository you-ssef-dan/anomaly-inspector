import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from PIL import Image

SIZE = 64


def write_image(path: Path, seed: int, mode: str = "RGB") -> None:
    """Write a small noise image; the seed makes every file unique, so every sha1 differs."""
    rng = np.random.default_rng(seed)
    shape = (SIZE, SIZE, 3) if mode == "RGB" else (SIZE, SIZE)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rng.integers(0, 256, shape, dtype=np.uint8), mode=mode).save(path)


def write_mask(path: Path) -> None:
    mask = np.zeros((SIZE, SIZE), dtype=np.uint8)
    mask[10:20, 10:20] = 255
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).save(path)


@pytest.fixture
def fake_mvtec(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A tiny MVTec-like dataset. Returns the relative root, as the real pipeline uses it.

    bottle (RGB): 4 train/good, 2 test/good, 2 test/broken_large with masks.
    screw (grayscale): 3 train/good, 1 test/good, 1 test/scratch_head with a mask.
    """
    monkeypatch.chdir(tmp_path)
    root = Path("data/raw/mvtec_ad")
    layout = {
        "bottle": ("RGB", {"train/good": 4, "test/good": 2, "test/broken_large": 2}),
        "screw": ("L", {"train/good": 3, "test/good": 1, "test/scratch_head": 1}),
    }
    seed = 0
    for category, (mode, folders) in layout.items():
        for folder, count in folders.items():
            for i in range(count):
                seed += 1
                write_image(root / category / folder / f"{i:03d}.png", seed, mode)
                defect = folder.split("/")[1]
                if folder.startswith("test/") and defect != "good":
                    write_mask(root / category / "ground_truth" / defect / f"{i:03d}_mask.png")
    return root


@pytest.fixture
def fake_project(fake_mvtec: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The repo layout around the fake dataset, ready for the PatchCore stages and scripts.

    A manifest, a DVC pointer file, a params.yaml with a tiny setup (ResNet-18, 64 px) and a git
    repo with one commit. The backbone gets random weights, so nothing is downloaded.
    """
    from anomaly_inspector.data.manifest import build_manifest
    from anomaly_inspector.data.splits import assign_splits
    from anomaly_inspector.models.backbones import build_backbone

    # Bottle is split as usual. Screw has too few fake images to split, so all of its rows
    # count as train: they are there for code that must pick one category and skip the others.
    categories = ["bottle", "screw"]
    bottle = assign_splits(build_manifest(fake_mvtec, ["bottle"]), val_fraction=0.1, seed=42)
    screw = build_manifest(fake_mvtec, ["screw"]).assign(split="train")
    manifest = pd.concat([bottle, screw], ignore_index=True)
    Path("data/processed").mkdir(parents=True)
    manifest.to_parquet("data/processed/manifest.parquet", index=False)
    Path("data/raw/mvtec_ad.dvc").write_text("outs:\n- md5: abc123.dir\n  path: mvtec_ad\n")
    params = {
        "data": {"root": str(fake_mvtec), "categories": categories, "image_size": 64},
        "patchcore": {"backbone": "resnet18", "coreset_ratio": 0.5},
    }
    Path("params.yaml").write_text(yaml.safe_dump(params))
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@example.com"]
    for command in (["init", "-q"], ["add", "."], ["commit", "-q", "-m", "fake"]):
        subprocess.run([*git, *command], check=True, capture_output=True)

    monkeypatch.setattr(
        "anomaly_inspector.pipelines.train_patchcore.build_backbone",
        lambda name: build_backbone(name, pretrained=False),
    )
    return Path.cwd()
