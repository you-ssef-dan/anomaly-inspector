"""ingest: confirmed photos join the training data once each, in a commit DVC tracks."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from anomaly_inspector.pipelines import ingest


def write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content.encode())
    return path


@pytest.fixture
def raw(tmp_path: Path) -> Path:
    root = tmp_path / "raw"
    write(root / "bottle/train/good/000.png", "a known train photo")
    write(root / "bottle/test/broken_large/000.png", "a known test photo")
    write(root / "screw/train/good/000.png", "a screw")
    return root


def test_new_photos_are_copied_under_a_name_that_cannot_collide(raw: Path, tmp_path: Path) -> None:
    incoming = tmp_path / "incoming"
    new = write(incoming / "bottle/good/000.png", "a new photo")  # same name as a train photo

    assert ingest.copy_new_photos(incoming, raw, ["bottle"]) == [new]
    copied = raw / "bottle/train/good" / f"000_{ingest.sha1_of(new)[:8]}.png"
    assert copied.read_bytes() == new.read_bytes()
    assert (raw / "bottle/train/good/000.png").read_text() == "a known train photo"  # untouched


def test_an_image_already_in_the_category_is_skipped(raw: Path, tmp_path: Path) -> None:
    incoming = tmp_path / "incoming"
    write(incoming / "bottle/good/a.png", "a known test photo")  # already in test/
    first = write(incoming / "bottle/good/b.png", "new")
    write(incoming / "bottle/good/c.png", "new")  # the same photo twice in one batch

    assert ingest.copy_new_photos(incoming, raw, ["bottle"]) == [first]
    assert len(list((raw / "bottle/train/good").iterdir())) == 2


def test_only_the_categories_asked_for(raw: Path, tmp_path: Path) -> None:
    incoming = tmp_path / "incoming"
    write(incoming / "bottle/good/a.png", "for bottle")
    write(incoming / "screw/good/a.png", "for screw")

    assert [p.parent.parent.name for p in ingest.copy_new_photos(incoming, raw, ["screw"])] == [
        "screw"
    ]
    assert len(list((raw / "bottle/train/good").iterdir())) == 1


def test_the_new_data_is_committed_and_listed_last(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)
    for key in ("AUTHOR", "COMMITTER"):
        monkeypatch.setenv(f"GIT_{key}_NAME", "t")
        monkeypatch.setenv(f"GIT_{key}_EMAIL", "t@example.com")
    dvc = [sys.executable, "-m", "dvc"]
    subprocess.run(["git", "init", "-q"], check=True)
    subprocess.run([*dvc, "init", "-q"], check=True)
    write(repo / "data/raw/mvtec_ad/bottle/train/good/000.png", "known")
    subprocess.run([*dvc, "add", "-q", "data/raw/mvtec_ad"], check=True)
    subprocess.run(["git", "add", "-A"], check=True)
    subprocess.run(["git", "commit", "-qm", "start"], check=True)
    before = Path("data/raw/mvtec_ad.dvc").read_text()
    incoming = write(tmp_path / "incoming/bottle/good/7.png", "confirmed").parents[2]

    ingest.main(["--incoming", str(incoming), "--category", "bottle"])

    assert json.loads(capsys.readouterr().out.splitlines()[-1]) == ["bottle/good/7.png"]
    assert Path("data/raw/mvtec_ad.dvc").read_text() != before
    status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    assert status.stdout == ""  # committed: the retraining starts clean
    log = subprocess.run(["git", "log", "-1", "--format=%s"], capture_output=True, text=True)
    assert log.stdout.strip() == "data: 1 confirmed good photos for bottle"

    with pytest.raises(SystemExit) as stopped:  # the same photos again: nothing to retrain on
        ingest.main(["--incoming", str(incoming), "--category", "bottle"])
    assert stopped.value.code == ingest.NOTHING_NEW
