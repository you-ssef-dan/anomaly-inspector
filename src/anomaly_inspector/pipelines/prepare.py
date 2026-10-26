import json
from pathlib import Path

import pandas as pd

from anomaly_inspector.config import Params, load_params
from anomaly_inspector.data.manifest import build_manifest
from anomaly_inspector.data.splits import SPLITS, assign_splits
from anomaly_inspector.data.validation import manifest_schema, mask_problems

MANIFEST_PATH = Path("data/processed/manifest.parquet")
SUMMARY_PATH = Path("reports/data_summary.json")


def summarize(manifest: pd.DataFrame) -> dict[str, object]:
    """Image counts per category, per category and split, and per category and defect type."""
    per_split = (
        manifest.groupby(["category", "split"]).size().unstack(fill_value=0).reindex(columns=SPLITS)
    )
    per_defect = manifest.groupby(["category", "defect_type"]).size().unstack(fill_value=0)
    return {
        "images": len(manifest),
        "per_category": {str(c): int(n) for c, n in manifest["category"].value_counts().items()},
        "per_category_split": {
            str(c): {str(s): int(n) for s, n in row.fillna(0).items()}
            for c, row in per_split.iterrows()
        },
        "per_category_defect_type": {
            str(c): {str(d): int(n) for d, n in row.items() if n > 0}
            for c, row in per_defect.iterrows()
        },
    }


def prepare(
    params: Params, manifest_path: Path = MANIFEST_PATH, summary_path: Path = SUMMARY_PATH
) -> pd.DataFrame:
    """Build, validate and split the manifest. Nothing is written unless every check passes."""
    data = params.data
    manifest = build_manifest(data.root, data.categories)
    manifest_schema.validate(manifest, lazy=True)
    problems = mask_problems(manifest)
    if problems:
        raise ValueError("Invalid masks:\n" + "\n".join(problems))
    manifest = assign_splits(manifest, data.val_normal_fraction, data.seed)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(manifest_path, index=False)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summarize(manifest), indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    manifest = prepare(load_params())
    print(f"Wrote {len(manifest)} rows to {MANIFEST_PATH} and a summary to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
