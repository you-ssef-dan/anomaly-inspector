"""Inspect the factory line every 10 minutes: flagged parts wait in review/, the rest are archived.

The line PC's job as Airflow tasks. The photos the camera left in line/incoming/ go to the
inspection API over HTTP (Airflow computes nothing), in chunks scored in parallel with dynamic task
mapping. Every result is written to the inspection log first, one Parquet file per run under
inspection_log/date=<day>/. Only then is each photo moved: to review/ when the API flags it, for a
person to check, otherwise to archive/line/<day>/. A retry rewrites the same log file and skips the
photos already moved, so no result is lost or counted twice.

    list_new_images    the photos waiting, in chunks of 10 (none: the rest of the run is skipped)
    score_chunk[i]     one task per chunk, all at once: POST /v1/predict for each photo
    store_and_route    log every result, then move each photo
"""

import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from airflow.sdk import dag, task

WORKSPACE = Path(os.environ.get("INSPECTION_WORKSPACE", "/tmp/workspace"))
LINE = WORKSPACE / "line" / "incoming"
REVIEW = WORKSPACE / "review"
ARCHIVE = WORKSPACE / "archive" / "line"
LOG = WORKSPACE / "inspection_log"
API = "http://api:8000"  # on the shared inspection network
CHUNK = 10  # photos per score_chunk task


@dag(
    schedule="*/10 * * * *",
    start_date=datetime(2026, 11, 1),
    catchup=False,
    max_active_runs=1,  # two runs at once would score the same photos
    tags=["inspection"],
    doc_md=__doc__,
)
def batch_inspection() -> None:
    @task
    def list_new_images() -> list[list[str]]:
        """The photos waiting, by name. One still being copied ends in .part, so it waits."""
        names = sorted(p.name for p in LINE.glob("*.png"))
        print(f"{len(names)} photos on the line")
        return [names[i : i + CHUNK] for i in range(0, len(names), CHUNK)]

    @task(retries=3, retry_exponential_backoff=True)  # the API may be restarting
    def score_chunk(names: list[str]) -> list[dict[str, Any]]:
        """The API's answer for each photo. Names and scores go through XCom, never pixels."""
        import httpx

        results = []
        with httpx.Client(base_url=API, timeout=30) as client:
            for name in names:
                with (LINE / name).open("rb") as photo:
                    files = {"file": (name, photo, "image/png")}
                    response = client.post("/v1/predict", files=files)
                response.raise_for_status()
                result = response.json()
                del result["heatmap_png_base64"]  # not asked for: always empty
                results.append({**result, "scored_at": datetime.now(UTC).isoformat()})
        return results

    @task
    def store_and_route(chunks: list[list[dict[str, Any]]], dag_run: Any = None) -> None:
        """Log every result, and only then move each photo out of the line."""
        import pandas as pd

        results = [result for chunk in chunks for result in chunk]
        when = dag_run.run_after  # set for every run, scheduled or not
        log = pd.DataFrame(results).assign(run_id=dag_run.run_id)
        log["scored_at"] = pd.to_datetime(log["scored_at"])
        target = LOG / f"date={when:%Y-%m-%d}" / f"{when:%Y%m%dT%H%M%S}.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(f".{target.name}")  # readers of the folder skip dot files
        log.to_parquet(partial, index=False)
        partial.replace(target)

        archive = ARCHIVE / f"{when:%Y-%m-%d}"
        for result in results:
            photo = LINE / result["filename"]
            if photo.exists():  # a retry finds the photos it moved already gone
                folder = REVIEW if result["is_anomalous"] else archive
                folder.mkdir(parents=True, exist_ok=True)
                photo.replace(folder / photo.name)
        flagged = int(log["is_anomalous"].sum())
        print(f"{len(log)} photos logged in {target}: {flagged} to review/, the rest archived")

    store_and_route(score_chunk.expand(names=list_new_images()))


batch_inspection()
