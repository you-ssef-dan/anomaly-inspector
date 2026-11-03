"""Every day, compare the line's last 24 hours with the reference of the model in use.

Airflow orchestrates and never computes: the comparison runs in the training image, with the code
and the references of main, taken from Airflow's clone without a checkout (the retraining DAG may
be working in its working tree). Any alert fails the check task: that red task is the alert, and
reports/drift/<day>.md in the workspace says what moved.

    write_report    scores, brightness, contrast, sharpness and share flagged, against the
                    reference: the report, and as its last line the verdict
    check           fails when the verdict lists an alert
"""

import json
import os
from datetime import datetime
from pathlib import Path

from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sdk import Param, dag, task
from airflow.sdk.exceptions import AirflowFailException
from docker.types import Mount

WORKSPACE = Path(os.environ.get("INSPECTION_WORKSPACE", "/tmp/workspace"))
TRAIN_IMAGE = "anomaly-inspector-train:latest"  # make train-image

# The code of one commit of the clone (main unless the run says otherwise), without a checkout.
WRITE_REPORT = """
set -o pipefail
repo="$WORKSPACE/industrial-anomaly-detection"
git -C "$repo" fetch --quiet origin main || echo "GitHub out of reach: main as fetched last" >&2
mkdir -p /tmp/code
git -C "$repo" archive "{{ params.rev }}" src | tar -x -C /tmp/code
PYTHONPATH=/tmp/code/src python -m anomaly_inspector.pipelines.drift_report \\
    --workspace "$WORKSPACE" --repo "$repo" --rev "{{ params.rev }}" \\
    --until "{{ dag_run.run_after }}"
"""


@dag(
    schedule="@daily",
    start_date=datetime(2026, 11, 1),
    catchup=False,
    max_active_runs=1,
    params={
        "rev": Param(
            "origin/main",
            type="string",
            pattern=r"^[\w./-]+$",  # a commit or branch name, nothing for the shell to run
            description="The commit whose code and references to use",
        )
    },
    tags=["monitoring"],
    doc_md=__doc__,
)
def drift_report() -> None:
    dagshub = os.environ.get("DAGSHUB_TOKEN", "")  # a reference missing from the clone's cache
    write_report = DockerOperator(
        task_id="write_report",
        image=TRAIN_IMAGE,
        command=["bash", "-ec", WRITE_REPORT],
        # The log and the photos to read, the clone, and reports/drift/ to write.
        mounts=[Mount(source=str(WORKSPACE), target=str(WORKSPACE), type="bind")],
        user=f"{os.getuid()}:{os.getgid()}",
        environment={"WORKSPACE": str(WORKSPACE)},
        private_environment={"AWS_ACCESS_KEY_ID": dagshub, "AWS_SECRET_ACCESS_KEY": dagshub},
        mount_tmp_dir=False,  # Airflow itself runs in Docker
        auto_remove="success",
        do_xcom_push=True,  # its last line: the verdict
        cpus=2.0,
        mem_limit="2g",
    )

    @task
    def check(verdict: str) -> None:
        """Fail on any alert, without retrying: nothing will have changed a minute later."""
        result = json.loads(verdict)
        if result["alerts"]:
            raise AirflowFailException(f"{'; '.join(result['alerts'])}. See {result['report']}")
        print(f"No drift: {result['report']}")

    check(write_report.output)


drift_report()
