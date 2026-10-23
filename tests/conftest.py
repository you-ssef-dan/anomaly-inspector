from pathlib import Path

import numpy as np
import pytest
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
