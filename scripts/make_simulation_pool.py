"""Move a seeded fraction of each category's train/good images to a simulation pool.

The pool feeds the Phase 9 factory simulation, so training never sees it. Run it once,
before `dvc add`, and preview with --dry-run first.
"""

import argparse
import random
import shutil
from pathlib import Path

DEFAULT_CATEGORIES = ["bottle", "carpet", "screw"]


def plan_moves(
    raw_root: Path, pool_root: Path, category: str, fraction: float, seed: int
) -> tuple[list[tuple[Path, Path]], int]:
    """Return the (source, destination) pairs to move and the train/good size before the move."""
    pool_dir = pool_root / category / "good"
    if pool_dir.exists() and any(pool_dir.iterdir()):
        raise SystemExit(
            f"{pool_dir} is not empty: the pool already exists, refusing to move more."
        )
    train_dir = raw_root / category / "train" / "good"
    images = sorted(train_dir.glob("*.png"))  # sorted first, so the seed gives the same pick
    if not images:
        raise SystemExit(f"No images found in {train_dir}")
    chosen = sorted(random.Random(seed).sample(images, round(len(images) * fraction)))
    return [(path, pool_dir / path.name) for path in chosen], len(images)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw/mvtec_ad"))
    parser.add_argument("--pool-root", type=Path, default=Path("data/simulation"))
    parser.add_argument("--categories", nargs="+", default=DEFAULT_CATEGORIES)
    parser.add_argument("--fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dry-run", action="store_true", help="show what would move, change nothing"
    )
    args = parser.parse_args(argv)
    if not 0 < args.fraction < 1:
        parser.error("--fraction must be between 0 and 1")

    # Plan every category before moving anything, so a guard failure cannot leave a half-done pool.
    plans = {
        category: plan_moves(args.raw_root, args.pool_root, category, args.fraction, args.seed)
        for category in args.categories
    }
    verb = "would move" if args.dry_run else "moved"
    for category, (moves, total) in plans.items():
        if not args.dry_run:
            for source, destination in moves:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(source, destination)
        print(f"{category}: {verb} {len(moves)} of {total}, train/good keeps {total - len(moves)}")


if __name__ == "__main__":
    main()
