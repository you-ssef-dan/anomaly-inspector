"""Add the confirmed good photos to the training data: the ingest step of the retrain_patchcore DAG.

A person confirmed these parts good (the factory simulation stands in for them here), so they are
safe to learn from. Only those: training on the model's own decisions would teach it its mistakes.

Each photo in <incoming>/<category>/good/ is copied to data/raw/mvtec_ad/<category>/train/good/ as
<name>_<first 8 hex of its sha1>.png, unless the same image is already in that category (the
manifest refuses duplicates). Then `dvc add` and a commit: the retraining runs from a commit that
holds this data, so its MLflow run is clean and can be promoted.

    python -m anomaly_inspector.pipelines.ingest --incoming /incoming --category bottle

The last line lists the photos copied, as JSON paths relative to <incoming>, so the DAG can archive
them once a pull request carries them. With nothing new it exits with code 99: the DAG skips the
rest of the run.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

RAW = Path("data/raw/mvtec_ad")
POINTER = Path("data/raw/mvtec_ad.dvc")
NOTHING_NEW = 99


def sha1_of(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def copy_new_photos(incoming: Path, raw: Path, categories: Sequence[str]) -> list[Path]:
    """Copy each confirmed photo not yet in its category. Returns the incoming files copied."""
    copied = []
    for category in categories:
        known = {sha1_of(p) for p in (raw / category).glob("*/*/*.png")}  # every split
        target = raw / category / "train" / "good"
        for photo in sorted((incoming / category / "good").glob("*.png")):
            digest = sha1_of(photo)
            if digest in known:  # already trained or tested on, or twice in this batch
                continue
            known.add(digest)
            shutil.copyfile(photo, target / f"{photo.stem}_{digest[:8]}.png")
            copied.append(photo)
    return copied


def run(*command: str) -> None:
    subprocess.run(command, check=True)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--incoming", type=Path, required=True)
    parser.add_argument("--category", nargs="+", required=True)
    args = parser.parse_args(argv)

    copied = copy_new_photos(args.incoming, RAW, args.category)
    if not copied:
        print("Nothing new: every confirmed photo is already in the training data.")
        raise SystemExit(NOTHING_NEW)
    run(sys.executable, "-m", "dvc", "add", "--quiet", str(RAW))
    run("git", "add", str(POINTER))
    message = f"data: {len(copied)} confirmed good photos for {', '.join(args.category)}"
    run("git", "commit", "--quiet", "-m", message)
    print(f"{message}, committed")
    print(json.dumps([p.relative_to(args.incoming).as_posix() for p in copied]))


if __name__ == "__main__":
    main()
