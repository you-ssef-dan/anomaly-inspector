"""The two Airflow DAGs: they import, keep their shape, and their Python tasks do what they say.

They run in Airflow's own image, not in the project's environment (Airflow pins many libraries
the project does not use), with the repository mounted read-only:

    make test-dags        # what CI's airflow-dags job runs

In the project's environment, without Airflow, they are skipped.
"""

import json
import re
from collections.abc import Callable
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

# Not "airflow": without Airflow installed, that finds this repository's airflow/ folder.
pytest.importorskip("airflow.sdk")

# httpx and pandas come with Airflow's image, like the DAGs that use them.
import httpx
import pandas as pd
from airflow.dag_processing.dagbag import DagBag
from airflow.providers.docker.operators.docker import DockerOperator

DAGS = Path(__file__).parents[2] / "airflow" / "dags"


@pytest.fixture(scope="module")
def dagbag() -> DagBag:
    return DagBag(dag_folder=str(DAGS))


@pytest.fixture
def workspace(dagbag: DagBag, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """batch_inspection's folders, moved to a temporary workspace with an empty line."""
    task_function = dagbag.dags["batch_inspection"].get_task("list_new_images").python_callable
    folders = {
        "LINE": tmp_path / "line" / "incoming",
        "REVIEW": tmp_path / "review",
        "ARCHIVE": tmp_path / "archive" / "line",
        "LOG": tmp_path / "inspection_log",
    }
    for name, folder in folders.items():
        monkeypatch.setitem(task_function.__globals__, name, folder)
    folders["LINE"].mkdir(parents=True)
    return tmp_path


def batch_task(dagbag: DagBag, task_id: str) -> Callable[..., Any]:
    return dagbag.dags["batch_inspection"].get_task(task_id).python_callable


def test_both_dags_import_without_errors(dagbag: DagBag) -> None:
    assert dagbag.import_errors == {}
    assert set(dagbag.dags) == {"retrain_patchcore", "batch_inspection"}


def test_schedules_and_one_run_at_a_time(dagbag: DagBag) -> None:
    retrain, batch = dagbag.dags["retrain_patchcore"], dagbag.dags["batch_inspection"]
    assert (retrain.schedule, batch.schedule) == ("@weekly", "*/10 * * * *")
    for dag in (retrain, batch):
        assert not dag.catchup
        assert dag.max_active_runs == 1  # two runs would share the clone, or score the same photos


def test_retrain_runs_in_order_and_branches_on_the_result(dagbag: DagBag) -> None:
    dag = dagbag.dags["retrain_patchcore"]
    steps = ["new_data_ready", "start_branch", "ingest", "retrain", "register", "decide"]
    for step, next_step in pairwise(steps):
        assert next_step in dag.get_task(step).downstream_task_ids
    assert dag.get_task("decide").downstream_task_ids == {"open_pull_request", "no_change"}
    # Archive only what a pull request carries: after it, with the list ingest copied.
    assert dag.get_task("archive_incoming").upstream_task_ids == {"open_pull_request", "ingest"}
    assert dag.get_task("ingest").skip_on_exit_code == [99]  # ingest.NOTHING_NEW


@pytest.mark.parametrize(
    ("registered", "next_task"),
    [
        ({"bottle": True}, "open_pull_request"),
        ({"bottle": False, "screw": True}, "open_pull_request"),
        ({"bottle": False}, "no_change"),
    ],
)
def test_any_new_champion_opens_a_pull_request(
    dagbag: DagBag, registered: dict[str, bool], next_task: str
) -> None:
    decide = dagbag.dags["retrain_patchcore"].get_task("decide").python_callable
    assert decide(json.dumps(registered)) == next_task


def test_only_open_pull_request_gets_the_github_token(dagbag: DagBag) -> None:
    steps = [t for t in dagbag.dags["retrain_patchcore"].tasks if isinstance(t, DockerOperator)]
    assert len(steps) == 5
    for step in steps:
        assert step.image == "anomaly-inspector-train:latest"
        assert step.network_mode == "inspection"  # mlflow:5000
        assert {m["Target"]: m["ReadOnly"] for m in step.mounts} == {
            "/repo": False,
            "/incoming": True,  # the confirmed photos are never changed by a step
        }
        hidden = step._private_environment  # the provider's name: hidden in the UI and logs
        assert "AWS_SECRET_ACCESS_KEY" in hidden
        assert ("GH_TOKEN" in hidden) == (step.task_id == "open_pull_request")
        assert not {"GH_TOKEN", "AWS_SECRET_ACCESS_KEY"} & set(step.environment)


def test_new_photos_are_listed_in_chunks_and_partial_copies_wait(
    dagbag: DagBag, workspace: Path
) -> None:
    line = workspace / "line" / "incoming"
    for i in range(23):
        (line / f"{i:02d}.png").write_bytes(b"photo")
    (line / "23.png.part").write_bytes(b"half a photo")  # the simulator is still copying it

    chunks = batch_task(dagbag, "list_new_images")()

    assert [len(chunk) for chunk in chunks] == [10, 10, 3]
    assert chunks[0][0] == "00.png"


def test_a_chunk_is_scored_by_the_api(
    dagbag: DagBag, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (workspace / "line" / "incoming" / "a.png").write_bytes(b"photo")
    status = 200

    def api(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/predict"
        match = re.search(rb'filename="([^"]+)"', request.content)
        assert match is not None
        answer = {"filename": match[1].decode(), "is_anomalous": False, "heatmap_png_base64": None}
        return httpx.Response(status, json=answer)

    client = httpx.Client
    monkeypatch.setattr(
        httpx, "Client", lambda **kwargs: client(transport=httpx.MockTransport(api), **kwargs)
    )
    score_chunk = dagbag.dags["batch_inspection"].get_task("score_chunk")

    (result,) = score_chunk.python_callable(["a.png"])
    assert result["filename"] == "a.png"
    assert "heatmap_png_base64" not in result  # not asked for, always empty: not logged
    assert datetime.fromisoformat(result["scored_at"]).tzinfo is not None

    status = 503  # the API is restarting: the task fails, and Airflow retries it
    with pytest.raises(httpx.HTTPStatusError):
        score_chunk.python_callable(["a.png"])
    assert score_chunk.retries == 3


def test_results_are_logged_then_photos_moved_and_a_retry_changes_nothing(
    dagbag: DagBag, workspace: Path
) -> None:
    line = workspace / "line" / "incoming"
    for name in ("flagged.png", "passed.png"):
        (line / name).write_bytes(b"photo")
    common = {"category": "bottle", "threshold": 2.0, "model_version": "abc", "latency_ms": 5.0}
    chunks = [
        [{"filename": "flagged.png", "anomaly_score": 3.0, "is_anomalous": True, **common}],
        [{"filename": "passed.png", "anomaly_score": 1.0, "is_anomalous": False, **common}],
    ]
    for chunk in chunks:
        chunk[0]["scored_at"] = "2026-11-01T21:30:05+00:00"
    run = SimpleNamespace(run_after=datetime(2026, 11, 1, 21, 30, tzinfo=UTC), run_id="scheduled")
    store_and_route = batch_task(dagbag, "store_and_route")

    store_and_route(chunks, dag_run=run)
    store_and_route(chunks, dag_run=run)  # a retry, after the photos moved

    assert (workspace / "review" / "flagged.png").exists()
    assert (workspace / "archive" / "line" / "2026-11-01" / "passed.png").exists()
    assert list(line.iterdir()) == []
    log_files = [p for p in (workspace / "inspection_log").rglob("*") if p.is_file()]
    assert [p.relative_to(workspace).as_posix() for p in log_files] == [
        "inspection_log/date=2026-11-01/20261101T213000.parquet"
    ]
    log = pd.read_parquet(workspace / "inspection_log")
    assert sorted(log["filename"]) == ["flagged.png", "passed.png"]  # once each
    assert set(log["run_id"]) == {"scheduled"}
