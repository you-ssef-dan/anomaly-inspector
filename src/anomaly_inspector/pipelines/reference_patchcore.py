"""DVC stage reference_patchcore@<category>: the served model's answers when all is well.

  python -m anomaly_inspector.pipelines.reference_patchcore --category bottle

One row per val_normal and holdout image, images no model trained on: its score and decision
from model.onnx, through the same Inspector the API runs, and its brightness, contrast and
sharpness. Written to models/patchcore/<category>/reference.parquet, next to the model it
describes, so each model version has its own. The daily drift check compares the line's photos
with it; the columns it shares with the inspection log have the same names.
"""

import argparse
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
from PIL import Image

from anomaly_inspector.image_stats import image_stats
from anomaly_inspector.serving.inference import Inspector

MANIFEST_PATH = Path("data/processed/manifest.parquet")
# Never trained on. Not calib: the threshold was chosen on it, so its decisions look too good.
SPLITS = ["val_normal", "holdout"]
KEPT = ["path", "category", "split", "label", "defect_type"]  # from the manifest


def onnx_path(category: str) -> Path:
    return Path("models/patchcore") / category / "model.onnx"


def reference_path(category: str) -> Path:
    return Path("models/patchcore") / category / "reference.parquet"


def build_reference(images: pd.DataFrame, inspector: Inspector) -> pd.DataFrame:
    """Score and measure each image, one at a time, as the API would."""
    measured = []
    for path in images["path"]:
        with Image.open(path) as image:
            prediction = inspector.predict(image)
            measured.append(
                {
                    "anomaly_score": prediction.score,
                    "is_anomalous": prediction.is_anomalous,
                    **image_stats(image),
                }
            )
    reference = images[KEPT].reset_index(drop=True).join(pd.DataFrame(measured))
    return reference.assign(threshold=inspector.threshold, model_version=inspector.version)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", required=True)
    args = parser.parse_args(argv)

    manifest = pd.read_parquet(MANIFEST_PATH)
    images = manifest[(manifest["category"] == args.category) & manifest["split"].isin(SPLITS)]
    reference = build_reference(images, Inspector(onnx_path(args.category)))
    reference.to_parquet(reference_path(args.category), index=False)

    flagged = reference.groupby("label")["is_anomalous"].mean()
    print(
        f"{args.category}: {len(reference)} images in {reference_path(args.category)}; "
        f"flagged: {flagged.get(0, 0.0):.0%} of good parts, {flagged.get(1, 0.0):.0%} of defects"
    )


if __name__ == "__main__":
    main()
