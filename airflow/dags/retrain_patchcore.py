"""Retrain PatchCore on newly confirmed good photos, and propose better models in a pull request.

Airflow orchestrates and never computes: every ML step runs in the training image (DockerOperator)
on Airflow's own clone of the repository. And Airflow never deploys: a new champion becomes a pull
request, CI checks it with the quality gate, a person merges it, and a release deploys it.

    new_data_ready     at least 20 confirmed photos in a category? Otherwise the run stops here.
    start_branch       a retrain/<time> branch from origin/main, with main's data and models
    ingest             the new photos join the training data, in a commit (so the runs are clean)
    retrain            dvc repro
    register           MLflow: each retrained category is the challenger, maybe the champion
    decide             a new champion -> open_pull_request -> archive_incoming
                       none           -> no_change (the photos wait in incoming/ for next time)
"""

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sdk import dag, task
from docker.types import Mount

WORKSPACE = Path(os.environ.get("INSPECTION_WORKSPACE", "/tmp/workspace"))
REPO = WORKSPACE / "industrial-anomaly-detection"
INCOMING = WORKSPACE / "incoming"
CATEGORIES = ["bottle", "carpet", "screw"]
MIN_NEW_PHOTOS = 20
TRAIN_IMAGE = "anomaly-inspector-train:latest"  # make train-image
CPUS = 8  # of this laptop's 18; PyTorch is told the same, or it starts a thread per core
READY = "{{ ti.xcom_pull(task_ids='new_data_ready') | join(' ') }}"

START_BRANCH = """
git fetch --quiet origin main
git checkout --quiet --force -B "retrain/{{ ts_nodash }}" origin/main
if [ "$(sha256sum < uv.lock)" != "$(cat /opt/uv.lock.sha256)" ]; then
  echo "The training image was built from another uv.lock than main's: run make train-image" >&2
  exit 1
fi
dvc pull --quiet --force
"""


def in_train_image(
    task_id: str, command: str, *, github: bool = False, **kwargs: Any
) -> DockerOperator:
    """A step in the training image, on the clone, as the clone's owner (not the image's user).

    The tokens come from the scheduler's environment (the only Airflow container that has them),
    and the GitHub one goes only to the steps that ask for it.
    """
    dagshub = os.environ.get("DAGSHUB_TOKEN", "")
    secrets = {"AWS_ACCESS_KEY_ID": dagshub, "AWS_SECRET_ACCESS_KEY": dagshub}
    if github:
        secrets["GH_TOKEN"] = os.environ.get("GH_TOKEN", "")
    kwargs.setdefault("do_xcom_push", False)  # only ingest and register pass a result on
    return DockerOperator(
        task_id=task_id,
        image=TRAIN_IMAGE,
        command=["bash", "-ec", command],
        mounts=[
            Mount(source=str(REPO), target="/repo", type="bind"),
            Mount(source=str(INCOMING), target="/incoming", type="bind", read_only=True),
        ],
        user=f"{os.getuid()}:{os.getgid()}",
        network_mode="inspection",  # mlflow:5000
        environment={
            "MLFLOW_TRACKING_URI": "http://mlflow:5000",
            "OMP_NUM_THREADS": str(CPUS),
            "GIT_AUTHOR_NAME": "retrain_patchcore DAG",
            "GIT_AUTHOR_EMAIL": "airflow@anomaly-inspector.invalid",
            "GIT_COMMITTER_NAME": "retrain_patchcore DAG",
            "GIT_COMMITTER_EMAIL": "airflow@anomaly-inspector.invalid",
        },
        private_environment=secrets,  # hidden in the UI and the logs
        mount_tmp_dir=False,  # Airflow itself runs in Docker
        auto_remove="success",
        cpus=float(CPUS),  # the default is 1.0
        mem_limit="6g",
        **kwargs,
    )


@dag(
    schedule="@weekly",
    start_date=datetime(2026, 11, 1),
    catchup=False,
    max_active_runs=1,  # one DVC pipeline at a time on the clone
    tags=["retraining"],
    doc_md=__doc__,
)
def retrain_patchcore() -> None:
    @task.short_circuit
    def new_data_ready() -> list[str]:
        """The categories with enough confirmed photos. None: the rest of the run is skipped."""
        counts = {c: len(list((INCOMING / c / "good").glob("*.png"))) for c in CATEGORIES}
        print(f"Confirmed photos waiting: {counts} (a retrain needs {MIN_NEW_PHOTOS})")
        return [c for c, n in counts.items() if n >= MIN_NEW_PHOTOS]

    start_branch = in_train_image("start_branch", START_BRANCH)
    ingest = in_train_image(
        "ingest",
        f"python -m anomaly_inspector.pipelines.ingest --incoming /incoming --category {READY}",
        skip_on_exit_code=99,  # nothing new after all
        do_xcom_push=True,  # its last line: the photos copied
    )
    retrain = in_train_image("retrain", "dvc repro")
    register = in_train_image(
        "register",
        f"python -m anomaly_inspector.pipelines.register --category {READY}",
        do_xcom_push=True,  # its last line, like {"bottle": true}
    )

    @task.branch
    def decide(promoted: str) -> str:
        return "open_pull_request" if any(json.loads(promoted).values()) else "no_change"

    open_pull_request = in_train_image(
        "open_pull_request",
        "bash scripts/open_retrain_pr.sh '{{ ti.xcom_pull(task_ids=\"register\") }}'",
        github=True,  # pushes the branch and opens the pull request
    )

    @task
    def archive_incoming(copied: str, ts_nodash: str | None = None) -> None:
        """Move the photos the pull request carries out of incoming/: they are not counted twice."""
        archive = WORKSPACE / "archive" / "confirmed" / str(ts_nodash)
        for name in json.loads(copied):
            (archive / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.move(INCOMING / name, archive / name)
        print(f"Archived {len(json.loads(copied))} photos in {archive}")

    @task
    def no_change() -> None:
        print(
            "No challenger beat its champion: nothing to propose, nothing deployed. "
            "The confirmed photos stay in incoming/ and count again next time."
        )

    new_data_ready() >> start_branch >> ingest >> retrain >> register
    decide(register.output) >> [open_pull_request, no_change()]
    open_pull_request >> archive_incoming(ingest.output)


retrain_patchcore()
