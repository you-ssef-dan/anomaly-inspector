"""simulate_line.py: photos reach the line with their truth on record, and each pool photo is
confirmed good once at most."""

import csv
import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path(__file__).parents[2] / "scripts" / "simulate_line.py"
POOL = [f"{i:03d}.png" for i in range(5)]


@pytest.fixture(scope="module")
def simulate() -> ModuleType:
    spec = importlib.util.spec_from_file_location("simulate_line", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """5 pool photos, 2 defects and 1 good test photo, each with its own bytes."""
    root = tmp_path / "repo"
    files = [f"data/simulation/bottle/good/{name}" for name in POOL] + [
        "data/raw/mvtec_ad/bottle/test/broken_large/000.png",
        "data/raw/mvtec_ad/bottle/test/contamination/000.png",
        "data/raw/mvtec_ad/bottle/test/good/000.png",  # evaluation data: never on the line
    ]
    for file in files:
        (root / file).parent.mkdir(parents=True, exist_ok=True)
        (root / file).write_bytes(file.encode())
    return root


def run(simulate: ModuleType, workspace: Path, repo: Path, *args: str) -> None:
    simulate.main(["--workspace", str(workspace), "--repo", str(repo), "--interval", "0", *args])


def labels(workspace: Path) -> list[dict[str, str]]:
    with (workspace / "line" / "labels.csv").open(newline="") as f:
        return list(csv.DictReader(f))


def test_photos_reach_the_line_with_their_truth_on_record(
    simulate: ModuleType, repo: Path, tmp_path: Path
) -> None:
    ws = tmp_path / "ws"
    run(simulate, ws, repo, "--ticks", "1", "--batch", "12", "--confirm", "0", "--seed", "1")

    rows = labels(ws)
    on_line = sorted(p.name for p in (ws / "line" / "incoming").iterdir())
    assert on_line == sorted(row["file"] for row in rows)
    assert len(rows) == 12
    for row in rows:
        source = repo / row["source"]
        assert (ws / "line" / "incoming" / row["file"]).read_bytes() == source.read_bytes()
        assert row["truth"] == source.parent.name
        assert row["category"] == "bottle"
        assert row["truth"] not in row["file"]  # the name gives nothing away
    assert "data/raw/mvtec_ad/bottle/test/good/000.png" not in {row["source"] for row in rows}


@pytest.mark.parametrize(
    ("rate", "expected"), [("0", {"good"}), ("1", {"broken_large", "contamination"})]
)
def test_the_defect_rate_decides_good_or_defective(
    simulate: ModuleType, repo: Path, tmp_path: Path, rate: str, expected: set[str]
) -> None:
    ws = tmp_path / "ws"
    run(
        simulate,
        ws,
        repo,
        "--ticks",
        "1",
        "--batch",
        "20",
        "--confirm",
        "0",
        "--defect-rate",
        rate,
        "--seed",
        "0",
    )
    assert {row["truth"] for row in labels(ws)} == expected


def test_each_pool_photo_is_confirmed_good_once(
    simulate: ModuleType, repo: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ws = tmp_path / "ws"
    for _ in range(3):  # asks for 3 + 3 + 3 from a pool of 5
        run(simulate, ws, repo, "--ticks", "1", "--batch", "0", "--confirm", "3")

    confirmed = ws / "incoming" / "bottle" / "good"
    assert sorted(p.name for p in confirmed.iterdir()) == POOL
    ledger = (ws / "simulation" / "confirmed-bottle.txt").read_text().split()
    assert sorted(ledger) == POOL  # no name twice
    lines = capsys.readouterr().out.splitlines()
    assert [line.split("confirmed good: ")[1] for line in lines] == [
        "3",
        "2 (pool used up)",
        "0 (pool used up)",
    ]
    assert (
        sorted(p.name for p in (repo / "data/simulation/bottle/good").iterdir()) == POOL
    )  # copied, not moved


def test_ticks_append_to_one_labels_file(simulate: ModuleType, repo: Path, tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    run(simulate, ws, repo, "--ticks", "2", "--batch", "4", "--confirm", "1")
    run(simulate, ws, repo, "--ticks", "1", "--batch", "4", "--confirm", "1")
    assert len(labels(ws)) == 12  # one header, 3 x 4 rows
    assert not list(ws.rglob("*.part"))  # every copy finished under its real name


def test_without_photos_it_says_to_pull_them(simulate: ModuleType, tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="dvc pull"):
        run(simulate, tmp_path / "ws", tmp_path / "empty", "--ticks", "1")


@pytest.mark.parametrize("bad", [["--defect-rate", "1.5"], ["--batch", "-1"], ["--interval", "-5"]])
def test_impossible_settings_are_refused(
    simulate: ModuleType, repo: Path, tmp_path: Path, bad: list[str]
) -> None:
    with pytest.raises(SystemExit) as stopped:
        run(simulate, tmp_path / "ws", repo, "--ticks", "1", *bad)
    assert stopped.value.code == 2  # argparse's usage error
