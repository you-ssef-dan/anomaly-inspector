"""The daily drift check: the line's last 24 hours against the reference of the model in use.

  python -m anomaly_inspector.pipelines.drift_report --workspace ~/airflow-workspace \
      --repo ~/airflow-workspace/industrial-anomaly-detection

Reads the inspection log the batch_inspection DAG writes, and measures each photo again with
image_stats (in review/ when it was flagged, archive/line/<day>/ otherwise). Then, per category,
against the reference (reference_patchcore) read from the repository at --rev, without a checkout,
a Kolmogorov-Smirnov test and the population stability index (PSI), each photo counted once:
- the anomaly scores of the photos the model passed, against the reference photos it passed: how
  good parts look to it. A light change that pushes parts over the threshold shows below;
- the brightness, contrast and sharpness of every photo, against every reference photo.
A signal has drifted only when both tests say so: with a few dozen reference photos, either one
alone raises false alarms. Last, the share of photos flagged, every photo counted, against what
the reference predicts for a line with 10% defects.

Writes <workspace>/reports/drift/<day>.md. The last line printed is the verdict as JSON, for the
DAG's check task, which fails when it lists any alert.
"""

import argparse
import hashlib
import io
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import dvc.api
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from PIL import Image
from scipy.stats import ks_2samp

from anomaly_inspector.image_stats import image_stats

STATS = ["brightness", "contrast", "sharpness"]
PSI_LIMIT = 0.25  # a common rule of thumb: above 0.25, a large shift
KS_ALPHA = 0.01  # and the KS test must find the two samples different at this level
LINE_DEFECT_RATE = 0.10  # the simulated line's (simulate_line.py --defect-rate)
FLAGGED_MARGIN = 0.10  # alert above the predicted share flagged plus 10 points
MIN_PHOTOS = 20  # fewer in 24 hours: too few to compare


@dataclass(frozen=True)
class Check:
    signal: str
    reference: str  # what the reference shows
    window: str  # what the last 24 hours show
    psi: float | None
    ks_p: float | None
    drifted: bool


def psi(reference: NDArray[np.float64], current: NDArray[np.float64]) -> float:
    """Population stability index on bins cut at the reference's quantiles: about 5 values of the
    smaller sample per bin, 10 bins at most, and half a value added to every bin. With a few dozen
    photos, a bin can stay empty by chance; counted as nearly 0, it alone would add about 2.
    Values beyond the reference's range fall in the end bins."""
    bins = max(2, min(10, len(reference) // 5, len(current) // 5))
    edges = np.quantile(reference, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    ref = (np.histogram(reference, edges)[0] + 0.5) / (len(reference) + 0.5 * bins)
    cur = (np.histogram(current, edges)[0] + 0.5) / (len(current) + 0.5 * bins)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def compare(signal: str, reference: pd.Series, current: pd.Series) -> Check:
    base = reference.to_numpy(dtype=np.float64)
    values = current.dropna().to_numpy(dtype=np.float64)  # NaN: the photo was moved since
    if len(values) < MIN_PHOTOS:
        return Check(signal, f"{np.median(base):.3g}", "too few photos", None, None, False)
    index, p = psi(base, values), float(ks_2samp(base, values).pvalue)
    drifted = index > PSI_LIMIT and p < KS_ALPHA
    return Check(signal, f"{np.median(base):.3g}", f"{np.median(values):.3g}", index, p, drifted)


def distinct(window: pd.DataFrame) -> pd.DataFrame:
    """Each photo once, by its bytes (by name if it was moved since). The simulated line sends
    its few pool photos again and again, and repeats make a test sure of a difference it cannot
    see. A real line never shows the same photo twice: there, this changes nothing."""
    return window.assign(key=window["sha1"].fillna(window["filename"])).drop_duplicates("key")


def check_category(
    window: pd.DataFrame, reference: pd.DataFrame, defect_rate: float
) -> list[Check]:
    """The photos of one category, scored by one model, against that model's reference."""
    photos = distinct(window)
    passed = photos[~photos["is_anomalous"].astype(bool)]
    reference_passed = reference[~reference["is_anomalous"]]
    checks = [compare("anomaly_score", reference_passed["anomaly_score"], passed["anomaly_score"])]
    checks += [compare(signal, reference[signal], photos[signal]) for signal in STATS]

    # What the reference's error rates predict for a line with this share of defects.
    good = reference[reference["label"] == 0]
    false_alarms = good["is_anomalous"].mean()
    caught = reference.loc[reference["label"] == 1, "is_anomalous"].mean()
    expected = (1 - defect_rate) * false_alarms + defect_rate * caught
    share = window["is_anomalous"].mean()
    too_high = share > expected + FLAGGED_MARGIN
    checks.append(Check("share flagged", f"{expected:.1%}", f"{share:.1%}", None, None, too_high))
    return checks


def read_log(workspace: Path, since: pd.Timestamp, until: pd.Timestamp) -> pd.DataFrame:
    """The photos scored in [since, until), with where batch_inspection moved each one."""
    first_day = f"{since - timedelta(days=1):%Y-%m-%d}"  # a run's folder is named by its start
    frames = []
    for path in sorted((workspace / "inspection_log").glob("date=*/*.parquet")):
        day = path.parent.name.removeprefix("date=")
        if path.name.startswith(".") or day < first_day:  # dot file: still being written
            continue
        frame = pd.read_parquet(path)
        frame["photo"] = [
            workspace / "review" / name if flagged else workspace / "archive" / "line" / day / name
            for name, flagged in zip(frame["filename"], frame["is_anomalous"], strict=True)
        ]
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["category", "model_version", "is_anomalous", "photo"])
    log = pd.concat(frames, ignore_index=True)
    scored = pd.to_datetime(log["scored_at"], utc=True)
    return log[(scored >= since) & (scored < until)].reset_index(drop=True)


def measure(log: pd.DataFrame) -> pd.DataFrame:
    """The log with each photo's SHA-1, brightness, contrast and sharpness: empty if it was
    moved since."""
    rows: list[dict[str, str | float]] = []
    for photo in log["photo"]:
        try:
            data = Path(photo).read_bytes()
        except FileNotFoundError:  # a person reviewed it already
            rows.append({})
            continue
        with Image.open(io.BytesIO(data)) as image:
            rows.append({"sha1": hashlib.sha1(data).hexdigest(), **image_stats(image)})
    return log.join(pd.DataFrame(rows, index=log.index, columns=["sha1", *STATS]))


def load_reference(repo: Path, rev: str, category: str) -> pd.DataFrame:
    """The reference at that commit, without a checkout: the retraining DAG may be working in
    the repository. From its DVC cache, or its DVC remote."""
    path = f"models/patchcore/{category}/reference.parquet"
    return pd.read_parquet(io.BytesIO(dvc.api.read(path, repo=str(repo), rev=rev, mode="rb")))


def markdown(since: pd.Timestamp, until: pd.Timestamp, sections: list[str]) -> str:
    rules = (
        "Each photo counts once: the anomaly scores of the photos the model passed, against the "
        "reference photos it passed; brightness, contrast and sharpness of all photos, against "
        f"all reference photos. A signal drifts when its PSI is above {PSI_LIMIT} and the KS "
        f"test's p-value below {KS_ALPHA}. The share flagged counts every photo the line sent, and "
        f"alerts more than {FLAGGED_MARGIN:.0%} above what the reference predicts."
    )
    title = f"# Drift report: {since:%Y-%m-%d %H:%M} to {until:%Y-%m-%d %H:%M} UTC"
    body = sections or ["No photo was inspected in these 24 hours."]
    return "\n\n".join([title, rules, *body]) + "\n"


def table(checks: list[Check]) -> str:
    rows = [
        "| Signal | Reference | Last 24 hours | PSI | KS p-value | Drift |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for c in checks:
        index = "" if c.psi is None else f"{c.psi:.2f}"
        p = "" if c.ks_p is None else f"{c.ks_p:.2g}"
        drift = "**yes**" if c.drifted else "no"
        rows.append(f"| {c.signal} | {c.reference} | {c.window} | {index} | {p} | {drift} |")
    return "\n".join(rows)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True, help="where the references are")
    parser.add_argument("--rev", default="origin/main", help="the commit to take them from")
    parser.add_argument("--until", type=pd.Timestamp, default=pd.Timestamp.now(tz="UTC"))
    parser.add_argument("--defect-rate", type=float, default=LINE_DEFECT_RATE)
    args = parser.parse_args(argv)
    until = args.until.tz_convert("UTC") if args.until.tzinfo else args.until.tz_localize("UTC")
    since = until - timedelta(hours=24)

    log = measure(read_log(args.workspace, since, until))
    sections, alerts = [], []
    for category, rows in log.groupby("category"):
        reference = load_reference(args.repo, args.rev, str(category))
        version = reference["model_version"].iloc[0]
        same = rows[rows["model_version"] == version]
        others = sorted(set(rows["model_version"]) - {version})
        heading = f"## {category}\n\n{len(rows)} photos; the reference describes model {version}."
        if others:
            left_out = len(rows) - len(same)
            heading += f" {left_out} scored by another model ({', '.join(others)}) are left out."
        if same.empty:
            alerts.append(f"{category}: no photo was scored by {args.rev}'s model")
            sections.append(
                f"{heading}\n\nNothing to compare. If the API still serves another model, "
                f"rebuild it from {args.rev}."
            )
            continue
        if len(same) < MIN_PHOTOS:
            sections.append(f"{heading}\n\nOnly {len(same)} photos: too few to compare.")
            continue
        checks = check_category(same, reference, args.defect_rate)
        alerts += [f"{category}: {c.signal}" for c in checks if c.drifted]
        compared = f"{len(same)} compared, {len(distinct(same))} of them distinct."
        sections.append(f"{heading} {compared}\n\n{table(checks)}")

    report = args.workspace / "reports" / "drift" / f"{until:%Y-%m-%d}.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(markdown(since, until, sections))
    print(report.read_text())
    print(json.dumps({"alerts": alerts, "report": str(report)}), flush=True)  # last: the verdict


if __name__ == "__main__":
    main()
