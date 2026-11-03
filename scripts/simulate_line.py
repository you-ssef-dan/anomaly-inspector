"""Simulate the factory: parts arrive on the line, and people confirm some good ones.

Every tick it copies:

- a batch of photos into <workspace>/line/incoming/, for the batch_inspection DAG: good parts
  from the simulation pool mixed with real defects from the test set. The file names give
  nothing away; the truth goes to <workspace>/line/labels.csv.
- a few pool photos into <workspace>/incoming/<category>/good/, as if a person had confirmed
  them good, for the retrain_patchcore DAG. Each pool photo is confirmed once at most: training
  refuses the same image twice. A confirmed photo never goes on the line again: the next model
  trains on it, and a real line never shows the same part twice. (The daily drift check caught
  it: the 19 bottle photos retraining had used scored lower than any new part.)

    python3 scripts/simulate_line.py                                    # every 5 min, until Ctrl+C
    python3 scripts/simulate_line.py --ticks 1 --batch 0 --confirm 20   # 20 confirmed photos now
    uv run python scripts/simulate_line.py --darken 0.3 --ticks 12      # an hour of dimmer light

Standard library only, so any Python 3.12 runs it, without the project's environment; only
--darken needs Pillow.
"""

import argparse
import csv
import os
import random
import shutil
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

LABEL_COLUMNS = ["file", "category", "truth", "source", "sent_at"]


def pool_photos(repo: Path, category: str) -> list[Path]:
    return sorted((repo / "data" / "simulation" / category / "good").glob("*.png"))


def confirmed_before(workspace: Path, category: str) -> set[str]:
    """The pool photos already confirmed good, from the ledger: they are training data now."""
    ledger = workspace / "simulation" / f"confirmed-{category}.txt"
    return set(ledger.read_text().split()) if ledger.exists() else set()


def defect_photos(repo: Path, category: str) -> list[Path]:
    test = repo / "data" / "raw" / "mvtec_ad" / category / "test"
    return sorted(p for p in test.glob("*/*.png") if p.parent.name != "good")


def copy_atomically(source: Path, target: Path, darken: float = 0.0) -> None:
    """Copy under a temporary name, then rename: a DAG listing *.png never sees half a file.
    With `darken`, every pixel loses that fraction of its value, as under a dimmer light."""
    partial = target.with_name(target.name + ".part")
    if darken:
        from PIL import Image  # this option only: the rest needs the standard library alone

        with Image.open(source) as image:
            image.point(lambda value: round(value * (1 - darken))).save(partial, format="PNG")
    else:
        shutil.copyfile(source, partial)
    os.replace(partial, target)


def send_to_line(
    workspace: Path,
    repo: Path,
    category: str,
    count: int,
    defect_rate: float,
    rng: random.Random,
    darken: float = 0.0,
) -> list[dict[str, str]]:
    """Put `count` photos on the line; each is a defect with probability `defect_rate`."""
    trained_on = confirmed_before(workspace, category)
    goods = [p for p in pool_photos(repo, category) if p.name not in trained_on]
    defects = defect_photos(repo, category)
    if count and not goods:
        raise SystemExit(
            f"Every {category} pool photo was confirmed good, so none is new for the line. "
            "Run it with --confirm 0, or start from an empty workspace/simulation/ ledger."
        )
    line = workspace / "line" / "incoming"
    line.mkdir(parents=True, exist_ok=True)
    sent_at = datetime.now(UTC)
    rows = []
    for _ in range(count):
        source = rng.choice(defects) if rng.random() < defect_rate else rng.choice(goods)
        rows.append(
            {
                "file": f"{sent_at:%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}.png",
                "category": category,
                "truth": source.parent.name,  # "good", or the defect type
                "source": source.relative_to(repo).as_posix(),
                "sent_at": sent_at.isoformat(timespec="seconds"),
            }
        )
    # The labels first, so every photo on the line has its truth on record.
    labels = workspace / "line" / "labels.csv"
    new_file = not labels.exists()
    with labels.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LABEL_COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        copy_atomically(repo / row["source"], line / row["file"], darken)
    return rows


def confirm_good(
    workspace: Path, repo: Path, category: str, count: int, rng: random.Random
) -> list[str]:
    """Copy up to `count` pool photos never confirmed before into incoming/<category>/good/."""
    ledger = workspace / "simulation" / f"confirmed-{category}.txt"
    done = confirmed_before(workspace, category)
    fresh = [p for p in pool_photos(repo, category) if p.name not in done]
    picked = sorted(rng.sample(fresh, min(count, len(fresh))))
    # The ledger first: a crash then loses a confirmation instead of repeating one.
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a") as f:
        f.writelines(f"{p.name}\n" for p in picked)
    target = workspace / "incoming" / category / "good"
    target.mkdir(parents=True, exist_ok=True)
    for p in picked:
        copy_atomically(p, target / p.name)
    return [p.name for p in picked]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--workspace", type=Path, default=Path.home() / "airflow-workspace")
    parser.add_argument(
        "--repo", type=Path, help="where the photos come from (default: the workspace's clone)"
    )
    parser.add_argument("--category", default="bottle", help="the one the API serves")
    parser.add_argument("--batch", type=int, default=10, help="photos on the line per tick")
    parser.add_argument("--defect-rate", type=float, default=0.1)
    parser.add_argument("--confirm", type=int, default=2, help="photos confirmed good per tick")
    parser.add_argument("--interval", type=float, default=300, help="seconds between ticks")
    parser.add_argument("--ticks", type=int, default=0, help="how many ticks; 0 = until Ctrl+C")
    parser.add_argument("--seed", type=int, help="the same seed picks the same photos")
    parser.add_argument(
        "--darken", type=float, default=0.0, help="0.3: the line's photos 30%% darker (Pillow)"
    )
    args = parser.parse_args(argv)
    if not 0 <= args.defect_rate <= 1:
        parser.error("--defect-rate must be between 0 and 1")
    if not 0 <= args.darken < 1:
        parser.error("--darken must be at least 0 and below 1")
    if min(args.batch, args.confirm, args.ticks) < 0 or args.interval < 0:
        parser.error("--batch, --confirm, --ticks and --interval cannot be negative")

    repo = args.repo or args.workspace / "industrial-anomaly-detection"
    if not pool_photos(repo, args.category) or not defect_photos(repo, args.category):
        raise SystemExit(
            f"No {args.category} pool or defect photos under {repo}/data: run `dvc pull` there."
        )

    rng = random.Random(args.seed)
    tick = 0
    try:
        while args.ticks == 0 or tick < args.ticks:
            if tick:
                time.sleep(args.interval)
            rows = send_to_line(
                args.workspace, repo, args.category, args.batch, args.defect_rate, rng, args.darken
            )
            confirmed = confirm_good(args.workspace, repo, args.category, args.confirm, rng)
            defective = sum(row["truth"] != "good" for row in rows)
            note = " (pool used up)" if len(confirmed) < args.confirm else ""
            darker = f", {args.darken:.0%} darker" if args.darken else ""
            print(
                f"{datetime.now():%H:%M:%S}  line: {len(rows)} photos, {defective} defective"
                f"{darker}  |  confirmed good: {len(confirmed)}{note}",
                flush=True,
            )
            tick += 1
    except KeyboardInterrupt:
        print("stopped")


if __name__ == "__main__":
    main()
