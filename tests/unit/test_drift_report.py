"""The daily drift check, on a made-up workspace: the inspection log batch_inspection writes, the
photos where it moved them, and a reference built from photos of the same made-up line."""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from anomaly_inspector.image_stats import image_stats
from anomaly_inspector.pipelines import drift_report
from anomaly_inspector.pipelines.drift_report import load_reference, measure, psi, read_log

UNTIL = pd.Timestamp("2026-11-03 00:00", tz="UTC")
TWO_HOURS_BEFORE = UNTIL - pd.Timedelta(hours=2)
VERSION = "abc1234"


def photo(seed: int, gain: float = 1.0) -> Image.Image:
    """A photo of the made-up line: smooth shapes plus noise. A gain below 1 darkens it."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:48, 0:48]
    shapes = 120 + 60 * np.sin(x / 5 + seed) * np.cos(y / 7)
    pixels = (shapes + rng.normal(0, 15, shapes.shape)) * gain
    return Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))


def reference(good: int = 30, defects: int = 10) -> pd.DataFrame:
    """Good parts score about 1.5 and pass; defects score about 3 and are all caught."""
    rng = np.random.default_rng(1)
    rows = []
    for i in range(good + defects):
        defect = i >= good
        score = rng.normal(3.0 if defect else 1.5, 0.2)
        stats = image_stats(photo(1000 + i))
        rows.append({"label": int(defect), "anomaly_score": score, "is_anomalous": defect, **stats})
    return pd.DataFrame(rows).assign(model_version=VERSION)


def line(
    workspace: Path,
    photos: int = 40,
    *,
    gain: float = 1.0,
    flag_every: int = 10,
    version: str = VERSION,
    at: pd.Timestamp = TWO_HOURS_BEFORE,
    distinct: int | None = None,
) -> None:
    """One batch_inspection run: its log file, and its photos in review/ or archive/line/<day>/.
    With `distinct`, only that many different photos, sent again and again."""
    rng = np.random.default_rng(2)
    day, rows = f"{at:%Y-%m-%d}", []
    for i in range(photos):
        flagged = i % flag_every == 0
        folder = workspace / "review" if flagged else workspace / "archive" / "line" / day
        folder.mkdir(parents=True, exist_ok=True)
        name = f"{at:%H%M%S}-{i:03d}.png"
        photo(i % (distinct or photos), gain).save(folder / name)
        rows.append(
            {
                "filename": name,
                "category": "bottle",
                "anomaly_score": rng.normal(3.0 if flagged else 1.5, 0.2),
                "threshold": 2.2,
                "is_anomalous": flagged,
                "model_version": version,
                "scored_at": at + pd.Timedelta(seconds=i),
            }
        )
    log = workspace / "inspection_log" / f"date={day}" / f"{at:%Y%m%dT%H%M%S}.parquet"
    log.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(log, index=False)


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An empty workspace; the reference comes straight from reference(), not from a repository."""
    monkeypatch.setattr(drift_report, "load_reference", lambda repo, rev, category: reference())
    return tmp_path


def check(workspace: Path, capsys: pytest.CaptureFixture[str]) -> dict[str, Any]:
    """Run the drift check; its verdict is the last line it prints."""
    args = ["--workspace", str(workspace), "--repo", "unused", "--until", str(UNTIL)]
    drift_report.main(args)
    verdict: dict[str, Any] = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    return verdict


def test_psi_is_small_for_the_same_distribution_and_large_for_a_shift() -> None:
    rng = np.random.default_rng(0)
    base = rng.normal(0, 1, 30)
    assert psi(base, rng.normal(0, 1, 3000)) < 0.25
    assert psi(base, rng.normal(2, 1, 3000)) > 1
    assert psi(base, np.full(100, 50.0)) > 1  # far beyond the reference: in the last bin


def test_a_bin_left_empty_by_chance_does_not_look_like_a_large_shift() -> None:
    base = np.linspace(0, 1, 30)
    current = np.linspace(0.25, 1, 25)  # the same spread, but nothing in the lowest bin
    assert psi(base, current) < 1  # counted as nearly 0, the empty bin alone would add 2.4


def test_a_normal_day_raises_no_alert_and_its_report_shows_every_signal(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    line(workspace)
    verdict = check(workspace, capsys)

    assert verdict == {"alerts": [], "report": str(workspace / "reports/drift/2026-11-03.md")}
    report = Path(verdict["report"]).read_text()
    assert "2026-11-02 00:00 to 2026-11-03 00:00 UTC" in report
    assert "40 photos; the reference describes model abc1234. 40 compared, 40 of them" in report
    for signal in ("anomaly_score", "brightness", "contrast", "sharpness", "share flagged"):
        assert f"| {signal} |" in report
    assert "| share flagged | 10.0% | 10.0% |" in report  # all defects caught, no false alarm


@pytest.mark.parametrize("flag_every", [10, 1])
def test_a_line_30_percent_darker_drifts_in_brightness_and_contrast(
    workspace: Path, capsys: pytest.CaptureFixture[str], flag_every: int
) -> None:
    # What phase 10's lighting test does. Every photo flagged (1): as the real bottle model does
    # with photos 30% darker, and they must still count for the image statistics.
    line(workspace, gain=0.7, flag_every=flag_every)
    alerts = check(workspace, capsys)["alerts"]
    assert {"bottle: brightness", "bottle: contrast"} <= set(alerts)
    assert "bottle: anomaly_score" not in alerts  # the made-up scores did not move


def test_too_many_photos_flagged_is_an_alert(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    line(workspace, flag_every=3)  # 34% flagged, 10% predicted
    # The scores compared are those of the photos the model passed: they did not move.
    assert check(workspace, capsys)["alerts"] == ["bottle: share flagged"]


def test_a_photo_sent_again_counts_once_in_the_tests(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    line(workspace, distinct=5)  # 40 photos on the line, but only 5 different ones
    verdict = check(workspace, capsys)
    report = Path(verdict["report"]).read_text()
    assert "40 compared, 5 of them distinct." in report
    for signal in ("anomaly_score", "brightness", "contrast", "sharpness"):
        assert f"| {signal} |" in report
        assert "too few photos" in report.split(f"| {signal} |")[1].splitlines()[0]
    assert "| share flagged | 10.0% | 10.0% |" in report  # every photo the line sent
    assert verdict["alerts"] == []


def test_photos_scored_by_another_model_are_not_compared(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    line(workspace, version="def5678")
    verdict = check(workspace, capsys)
    assert verdict["alerts"] == ["bottle: no photo was scored by origin/main's model"]
    report = Path(verdict["report"]).read_text()
    assert "40 scored by another model (def5678) are left out." in report
    assert "If the API still serves another model, rebuild it from origin/main." in report


def test_a_day_without_enough_photos_compares_nothing(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert check(workspace, capsys)["alerts"] == []
    assert "No photo was inspected" in (workspace / "reports/drift/2026-11-03.md").read_text()

    line(workspace, photos=10)
    assert check(workspace, capsys)["alerts"] == []
    assert "Only 10 photos: too few" in (workspace / "reports/drift/2026-11-03.md").read_text()


def test_only_the_24_hours_are_read_and_each_photo_is_found_where_it_was_moved(
    workspace: Path,
) -> None:
    line(workspace, photos=5, at=UNTIL - pd.Timedelta(hours=30))  # before the 24 hours
    line(workspace, photos=5, at=TWO_HOURS_BEFORE)
    log_folder = workspace / "inspection_log" / "date=2026-11-02"
    (log_folder / ".being-written.parquet").write_bytes(b"not parquet yet")

    log = read_log(workspace, UNTIL - pd.Timedelta(hours=24), UNTIL)
    assert len(log) == 5
    assert log["photo"].iloc[0] == workspace / "review" / "220000-000.png"  # flagged
    assert log["photo"].iloc[1] == workspace / "archive/line/2026-11-02/220000-001.png"

    log["photo"].iloc[1].unlink()  # a person moved it since
    measured = measure(log)
    assert measured["brightness"].isna().tolist() == [False, True, False, False, False]


def test_a_signal_with_too_few_photos_left_is_not_compared(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    line(workspace, photos=25)
    for path in list((workspace / "archive").rglob("*.png"))[:10]:
        path.unlink()  # moved since: no brightness for these 10
    report = Path(check(workspace, capsys)["report"]).read_text()
    assert "| brightness | " in report
    assert "too few photos" in report.split("| brightness |")[1].splitlines()[0]
    assert "too few photos" not in report.split("| anomaly_score |")[1].splitlines()[0]


def test_the_reference_is_read_at_a_commit_without_a_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """As in Airflow's clone: the working tree may hold a newer reference, from a retraining."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DVC_NO_ANALYTICS", "1")
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@example.com"]
    dvc = [sys.executable, "-m", "dvc"]
    path = Path("models/patchcore/bottle/reference.parquet")
    path.parent.mkdir(parents=True)
    committed = reference()
    committed.to_parquet(path, index=False)
    for command in (
        [*git, "init", "-q"],
        [*dvc, "init", "-q"],
        [*dvc, "add", "-q", str(path)],
        [*git, "add", "."],
        [*git, "commit", "-q", "-m", "a model and its reference"],
    ):
        subprocess.run(command, check=True, capture_output=True)
    reference(good=5).to_parquet(path, index=False)  # the working tree moves on

    loaded = load_reference(tmp_path, "HEAD", "bottle")
    pd.testing.assert_frame_equal(loaded, committed)
