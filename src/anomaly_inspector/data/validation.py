import numpy as np
import pandas as pd
import pandera.pandas as pa
from PIL import Image

manifest_schema = pa.DataFrameSchema(
    {
        "path": pa.Column(str, unique=True),
        "source_split": pa.Column(str, pa.Check.isin(["train", "test"])),
        "label": pa.Column(int, pa.Check.isin([0, 1])),
        "mask_path": pa.Column(str, nullable=True),
        "width": pa.Column(int, pa.Check.ge(64)),
        "sha1": pa.Column(str, unique=True),  # the same image twice = leakage
    },
    checks=[
        pa.Check(
            lambda df: (df.source_split != "train") | (df.label == 0),
            error="train must contain only good images",
        ),
        pa.Check(
            lambda df: (df.label == 0) | df.mask_path.notna(),
            error="every defective image needs a mask",
        ),
    ],
)


def mask_problems(manifest: pd.DataFrame) -> list[str]:
    """One message per mask with a size different from its image or values besides 0 and 255."""
    problems: list[str] = []
    for row in manifest[manifest["label"] == 1].itertuples():
        with Image.open(str(row.path)) as image, Image.open(str(row.mask_path)) as mask:
            if mask.size != image.size:
                problems.append(
                    f"{row.mask_path}: mask size {mask.size} != image size {image.size}"
                )
            values = set(np.unique(np.asarray(mask)).tolist())
            if not values <= {0, 255}:
                problems.append(f"{row.mask_path}: values {sorted(values)} are not only 0 and 255")
    return problems
