"""run_experiments.py and plot_experiments.py: the design of the experiments, and the charts."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from anomaly_inspector.data.manifest import build_manifest
from anomaly_inspector.data.splits import assign_splits
from anomaly_inspector.models.backbones import build_backbone

SCRIPTS = Path(__file__).parents[2] / "scripts"


def load_script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def experiments() -> ModuleType:
    return load_script("run_experiments")


@pytest.fixture(scope="module")
def plots() -> ModuleType:
    return load_script("plot_experiments")


@pytest.fixture
def manifest_path(fake_mvtec: Path) -> Path:
    manifest = assign_splits(build_manifest(fake_mvtec, ["bottle"]), val_fraction=0.1, seed=42)
    path = Path("manifest.parquet")
    manifest.to_parquet(path, index=False)
    return path


@pytest.fixture
def untrained_backbones(monkeypatch: pytest.MonkeyPatch, experiments: ModuleType) -> None:
    """Random weights, so no test downloads the 130 MB of ImageNet weights."""
    monkeypatch.setattr(
        experiments, "build_backbone", lambda name: build_backbone(name, pretrained=False)
    )


# --- the design of the experiments ----------------------------------------------------------


def test_plan_is_the_baseline_plus_one_change_per_run_and_one_combination(
    experiments: ModuleType,
) -> None:
    runs = experiments.plan()
    assert len(runs) == 19
    assert runs["baseline"] == experiments.BASELINE
    groups = {"images": {"images", "subset_seed"}}  # one change: which images, and the draw
    for name, settings in runs.items():
        changed = {k for k, v in settings.items() if v != experiments.BASELINE[k]}
        if name == "baseline":
            assert not changed
        elif name.startswith("images/"):
            assert changed == groups["images"]
        elif name.startswith("combo/"):
            assert changed == {"backbone", "size"}
        else:
            assert len(changed) == 1, name
    assert [n for n in runs if n.startswith("combo/")] == ["combo/resnet18-320"]


def test_every_planned_value_is_in_the_guides_experiments(experiments: ModuleType) -> None:
    settings = list(experiments.plan().values())
    assert {s["coreset_ratio"] for s in settings} == {0.001, 0.01, 0.1}
    assert {s["images"] for s in settings} == {"all", 1, 5, 10, 50}
    assert {s["backbone"] for s in settings} == {"wide_resnet50_2", "resnet18"}
    assert {s["size"] for s in settings} == {224, 256, 320}


def test_each_image_count_is_drawn_three_times(experiments: ModuleType) -> None:
    names = [n for n in experiments.plan() if n.startswith("images/")]
    assert sorted(names) == sorted(f"images/{n}/seed{s}" for n in (1, 5, 10, 50) for s in range(3))


def test_a_run_scores_the_calib_split_and_reports_cost(
    experiments: ModuleType, untrained_backbones: None, manifest_path: Path
) -> None:
    result = experiments.run("bottle", "baseline", manifest_path, batch_size=2)
    assert result["calib_images"] == 2  # holdout is not touched
    assert result["train_images"] == 3
    assert all(0 <= result[m] <= 1 for m in ("image_auroc", "pixel_auroc", "aupro"))
    assert result["memory_bank_rows"] == int(3 * 32 * 32 * 0.01)
    assert result["settings"] == experiments.BASELINE
    assert result["fit_seconds"] > 0
    assert result["peak_ram_mb"] > 0


def test_fewer_images_keep_the_memory_bank_the_size_of_the_baselines(
    experiments: ModuleType, untrained_backbones: None, manifest_path: Path
) -> None:
    baseline = experiments.run("bottle", "baseline", manifest_path, batch_size=2)
    one = experiments.run("bottle", "images/1/seed0", manifest_path, batch_size=2)
    assert one["train_images"] == 1
    assert one["memory_bank_rows"] == baseline["memory_bank_rows"]
    assert one["coreset_ratio_used"] > baseline["coreset_ratio_used"]


def test_a_bank_cannot_be_larger_than_the_patches_there_are(
    experiments: ModuleType,
    untrained_backbones: None,
    manifest_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A huge baseline bank, but one image only has 1,024 patches: the ratio stops at 1.
    monkeypatch.setitem(experiments.BASELINE, "coreset_ratio", 1.0)
    result = experiments.run("bottle", "images/1/seed0", manifest_path, batch_size=2)
    assert result["coreset_ratio_used"] == 1.0
    assert result["memory_bank_rows"] == 32 * 32


@pytest.fixture
def datasets_built(monkeypatch: pytest.MonkeyPatch, experiments: ModuleType) -> list[Any]:
    """Every (rows, size) a run builds a dataset from: training first, then the scored images."""
    built: list[Any] = []
    real = experiments.ManifestDataset

    def spy(rows: Any, size: int) -> Any:
        built.append((rows.copy(), size))
        return real(rows, size)

    monkeypatch.setattr(experiments, "ManifestDataset", spy)
    return built


def test_the_model_learns_from_train_and_is_scored_on_calib_only(
    experiments: ModuleType,
    untrained_backbones: None,
    manifest_path: Path,
    datasets_built: list[Any],
) -> None:
    experiments.run("bottle", "baseline", manifest_path, batch_size=2)
    (train, _), (scored, _) = datasets_built
    assert set(train["split"]) == {"train"}
    assert set(scored["split"]) == {"calib"}  # holdout and val_normal are never touched


def test_a_subset_is_drawn_from_its_seed_and_the_same_seed_gives_the_same_subset(
    experiments: ModuleType,
    untrained_backbones: None,
    manifest_path: Path,
    datasets_built: list[Any],
) -> None:
    def chosen(seed: int) -> list[str]:
        datasets_built.clear()
        experiments.run("bottle", f"images/1/seed{seed}", manifest_path, batch_size=2)
        return list(datasets_built[0][0]["path"])

    by_seed = {seed: chosen(seed) for seed in (0, 1, 2)}
    assert all(len(paths) == 1 for paths in by_seed.values())
    assert len({paths[0] for paths in by_seed.values()}) > 1  # the seed changes the image
    assert chosen(1) == by_seed[1]  # and the same seed always gives the same one


@pytest.mark.parametrize(
    ("name", "size"), [("baseline", 256), ("size/224", 224), ("size/320", 320)]
)
def test_the_model_sees_the_size_the_run_asks_for(
    experiments: ModuleType,
    untrained_backbones: None,
    manifest_path: Path,
    datasets_built: list[Any],
    name: str,
    size: int,
) -> None:
    experiments.run("bottle", name, manifest_path, batch_size=2)
    assert [s for _, s in datasets_built] == [size, size]  # training images and scored images


def test_finished_runs_are_not_started_again(
    experiments: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    out = tmp_path / "results.json"
    out.write_text(json.dumps({f"bottle/{name}": {} for name in experiments.plan()}))

    def fail(*_: Any, **__: Any) -> None:
        raise AssertionError("a finished run was started again")

    monkeypatch.setattr(subprocess, "run", fail)
    experiments.run_all("bottle", out, [])  # would raise if it started anything


def test_unfinished_runs_are_started_in_their_own_process(
    experiments: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    out = tmp_path / "results.json"
    out.write_text(json.dumps({"bottle/baseline": {}}))
    started: list[str] = []
    monkeypatch.setattr(
        subprocess, "run", lambda command, **_: started.append(command[command.index("--run") + 1])
    )
    experiments.run_all("bottle", out, ["--batch-size", "2"])
    assert started == [name for name in experiments.plan() if name != "baseline"]


# --- the charts and the table ---------------------------------------------------------------


@pytest.fixture
def results(experiments: ModuleType) -> dict[str, Any]:
    """Made-up numbers for every run of two categories, to exercise the plotting code."""
    out = {}
    for category in ("bottle", "screw"):
        for i, name in enumerate(experiments.plan()):
            out[f"{category}/{name}"] = {
                "image_auroc": 0.80 + 0.01 * (i % 5),
                "pixel_auroc": 0.90,
                "aupro": 0.70 + 0.01 * i,
                "ms_per_image": 100.0 + i,
                "memory_bank_mb": 10.0,
                "fit_seconds": 60.0,
            }
    return out


def test_every_experiment_chart_includes_the_baseline(
    plots: ModuleType, results: dict[str, Any]
) -> None:
    labels = {e: [x for x, _ in plots.series(results, "bottle", e)] for e in plots.TITLES}
    assert labels["coreset"] == ["0.1%", "1%", "10%"]
    assert labels["images"] == ["1", "5", "10", "50", "all"]
    assert labels["backbone"] == ["ResNet-18", "WideResNet-50"]
    assert labels["size"] == ["224 px", "256 px", "320 px"]


def test_image_counts_show_all_three_subsets_and_the_rest_one_run(
    plots: ModuleType, results: dict[str, Any]
) -> None:
    sizes = [len(runs) for _, runs in plots.series(results, "screw", "images")]
    assert sizes == [3, 3, 3, 3, 1]
    assert {len(runs) for _, runs in plots.series(results, "screw", "size")} == {1}


def test_a_missing_run_leaves_a_gap_instead_of_failing(
    plots: ModuleType, results: dict[str, Any]
) -> None:
    del results["screw/size/320"]
    assert [x for x, _ in plots.series(results, "screw", "size")] == ["224 px", "256 px"]
    assert plots.series({}, "screw", "size") == []


def test_the_table_has_one_row_per_run_with_subsets_averaged(
    plots: ModuleType, results: dict[str, Any]
) -> None:
    rows = [line for line in plots.table(results).splitlines() if line.startswith("| bottle")]
    # baseline, coreset, image counts, backbone, sizes, the combination
    assert len(rows) == 1 + 2 + 4 + 1 + 2 + 1
    assert rows[0].startswith("| bottle | baseline |")  # the baseline comes first
    assert any("combo/resnet18-320" in row for row in rows)


def test_charts_are_written_for_both_themes(
    plots: ModuleType, results: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(plots, "FIGURES", tmp_path)
    paths = [plots.plot(results, e, t) for e in plots.TITLES for t in plots.THEMES]
    assert len(paths) == 8
    assert all(p.exists() and p.stat().st_size > 5_000 for p in paths)
    assert {p.name for p in paths} == {
        f"experiment_{e}_{t}.png" for e in plots.TITLES for t in ("light", "dark")
    }


def test_the_validated_palette_is_used(plots: ModuleType) -> None:
    light, dark = plots.THEMES["light"]["series"], plots.THEMES["dark"]["series"]
    assert (light["bottle"], light["screw"]) == ("#2a78d6", "#eb6834")
    assert (dark["bottle"], dark["screw"]) == ("#3987e5", "#d95926")
