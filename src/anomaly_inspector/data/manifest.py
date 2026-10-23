import hashlib
from pathlib import Path

import pandas as pd
from PIL import Image

COLUMNS = [
    "path",
    "category",
    "source_split",
    "defect_type",
    "label",
    "mask_path",
    "width",
    "height",
    "mode",
    "sha1",
]


def build_manifest(root: Path, categories: list[str]) -> pd.DataFrame:
    """One row per train/test image. Paths keep the form of `root`, so it must be relative."""
    if root.is_absolute():
        raise ValueError(f"root must be relative to the repo root, got {root}")
    rows: list[dict[str, object]] = []
    for category in categories:
        for path in sorted((root / category).glob("*/*/*.png")):
            source_split, defect_type = path.parts[-3], path.parts[-2]
            if source_split == "ground_truth":
                continue
            mask = root / category / "ground_truth" / defect_type / f"{path.stem}_mask.png"
            with Image.open(path) as im:
                width, height = im.size
                mode = im.mode
            rows.append(
                {
                    "path": path.as_posix(),
                    "category": category,
                    "source_split": source_split,
                    "defect_type": defect_type,
                    "label": int(defect_type != "good"),
                    "mask_path": mask.as_posix() if mask.exists() else None,
                    "width": width,
                    "height": height,
                    "mode": mode,
                    "sha1": hashlib.sha1(path.read_bytes(), usedforsecurity=False).hexdigest(),
                }
            )
    return pd.DataFrame(rows, columns=COLUMNS)
