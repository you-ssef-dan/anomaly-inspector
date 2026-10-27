"""make_gallery.py: the whole script on the fake dataset (no download)."""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from anomaly_inspector.data.manifest import build_manifest
from anomaly_inspector.data.splits import assign_splits

SCRIPT = Path(__file__).parents[2] / "scripts" / "make_gallery.py"
ONE_MB = 1_000_000  # the pre-commit hook refuses larger files


@pytest.fixture(scope="module")
def gallery() -> ModuleType:
    spec = importlib.util.spec_from_file_location("make_gallery", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["make_gallery"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def manifest_path(fake_mvtec: Path) -> Path:
    manifest = assign_splits(build_manifest(fake_mvtec, ["bottle"]), val_fraction=0.1, seed=42)
    path = Path("manifest.parquet")
    manifest.to_parquet(path, index=False)
    return path


def arguments(*extra: str) -> list[str]:
    return [
        "--category=bottle",
        "--manifest=manifest.parquet",
        "--out-dir=out",
        "--json=out/gallery.json",
        "--backbone=resnet18",
        "--no-pretrained",
        "--size=64",
        "--coreset-ratio=0.5",
        "--k=1",
        *extra,
    ]


def test_the_script_writes_the_picture_and_the_facts_behind_it(
    gallery: ModuleType, manifest_path: Path
) -> None:
    gallery.main(arguments())
    assert 5_000 < Path("out/bottle.jpg").stat().st_size < ONE_MB

    entry = json.loads(Path("out/gallery.json").read_text())["bottle"]
    assert entry["calib_images"] == 2
    assert all(0 <= entry[m] <= 1 for m in ("image_auroc", "pixel_auroc", "aupro"))
    missed, rejected = entry["panels"]
    assert (missed["kind"], rejected["kind"]) == ("missed", "false_reject")
    assert missed["defect_type"] == "broken_large"
    assert rejected["defect_type"] == "good"
    assert missed["defect_percent"] == pytest.approx(100 * 100 / 64**2)  # a 10 x 10 square
    assert rejected["defect_percent"] is None
    assert isinstance(missed["peak_on_defect"], bool)
    assert rejected["peak_on_defect"] is None
    assert missed["rank_in_class"] == 1  # the only defective calib image of the fake dataset
    assert missed["images_in_class"] == rejected["images_in_class"] == 1


def test_the_top_row_holds_outlined_defects_and_the_bottom_row_good_parts(
    gallery: ModuleType, manifest_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    drawn: list[object] = []
    real = gallery.draw_gallery

    def spy(rows: object, *args: object, **kwargs: object) -> Path:
        drawn.append(rows)
        return real(rows, *args, **kwargs)  # type: ignore[no-any-return]

    monkeypatch.setattr(gallery, "draw_gallery", spy)
    gallery.main(arguments())

    top, bottom = drawn[0]  # type: ignore[misc]
    assert all(p.mask is not None and p.caption.startswith("broken_large") for p in top)
    assert all(p.mask is None and p.caption.startswith("good part") for p in bottom)
    assert all("lowest 1 of 1" in p.caption for p in top)
    assert all("highest 1 of 1" in p.caption for p in bottom)
    assert all(p.heat.min() >= 0 and p.heat.max() <= 1 for p in (*top, *bottom))
    assert all(p.image.shape == (64, 64, 3) for p in (*top, *bottom))


def test_the_picked_images_are_calib_images_never_holdout(
    gallery: ModuleType, manifest_path: Path
) -> None:
    gallery.main(arguments())
    import pandas as pd

    manifest = pd.read_parquet("manifest.parquet").set_index("path")
    shown = [
        p["path"] for p in json.loads(Path("out/gallery.json").read_text())["bottle"]["panels"]
    ]
    assert set(manifest.loc[shown, "split"]) == {"calib"}


def test_other_categories_already_in_the_file_are_kept(
    gallery: ModuleType, manifest_path: Path
) -> None:
    Path("out").mkdir()
    Path("out/gallery.json").write_text(json.dumps({"carpet": {"aupro": 0.9}}))
    gallery.main(arguments())
    assert set(json.loads(Path("out/gallery.json").read_text())) == {"bottle", "carpet"}
