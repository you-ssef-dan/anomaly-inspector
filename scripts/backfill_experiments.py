"""Send the Phase 2 experiments (reports/experiments.json) to MLflow, to compare them in the UI.

Run it from the repo root, with the MLflow server up (make mlflow-up):

  uv run --env-file .env python scripts/backfill_experiments.py

Each result becomes one run of the "phase2-experiments" experiment, named like its JSON key
(bottle/coreset/0.1). The runs are labelled as backfilled: their git SHA and data hash are those
of the commit that holds the results, not today's. Their accuracy was scored on calib, so those
metrics are named calib_*, apart from the holdout_* metrics that promotion compares. Nothing is
registered. Running it again adds only the results that are not in MLflow yet.
"""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import mlflow
import yaml

from anomaly_inspector.tracking import DATA_POINTER, git, tracking_uri

RESULTS = Path("reports/experiments.json")
EXPERIMENT = "phase2-experiments"
SCORED_ON_CALIB = {"image_auroc", "pixel_auroc", "aupro"}


def source_commit(path: Path) -> str:
    """The last commit that changed the results file: it holds them and the code that made them."""
    commit = git("log", "-1", "--format=%H", "--", str(path))
    if not commit:
        raise ValueError(f"{path} is not committed, so its results cannot be traced to code")
    return commit


def data_md5_at(commit: str) -> str:
    """The DVC data hash as it was in that commit (the data the experiments used)."""
    pointer = yaml.safe_load(git("show", f"{commit}:{DATA_POINTER.as_posix()}"))
    return str(pointer["outs"][0]["md5"])


def to_run(
    name: str, result: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, float], dict[str, str]]:
    """One JSON result as the params, metrics and tags of a run."""
    category, experiment = name.split("/")[:2]
    # The settings the run changed; subset_seed is None outside the image-count experiment.
    params = {key: value for key, value in result["settings"].items() if value is not None}
    params["coreset_ratio_used"] = result["coreset_ratio_used"]
    metrics = {
        f"calib_{key}" if key in SCORED_ON_CALIB else key: float(value)
        for key, value in result.items()
        if key not in ("settings", "coreset_ratio_used")
    }
    return params, metrics, {"category": category, "experiment": experiment}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--results", type=Path, default=RESULTS)
    args = parser.parse_args(argv)

    tracking_uri()  # refuses to go on without a server, as every run does
    results: dict[str, dict[str, Any]] = json.loads(args.results.read_text())
    commit = source_commit(args.results)
    common = {
        "backfilled_from": args.results.as_posix(),
        "git_sha": commit,
        "data_md5": data_md5_at(commit),
    }

    mlflow.set_experiment(EXPERIMENT)
    runs = mlflow.search_runs(experiment_names=[EXPERIMENT], output_format="list")
    done = {run.info.run_name for run in runs}
    added = 0
    for name, result in results.items():
        if name in done:
            continue
        params, metrics, tags = to_run(name, result)
        with mlflow.start_run(run_name=name, tags=common | tags):
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
        added += 1
    print(f"{EXPERIMENT}: {added} runs added, {len(results) - added} were already there")


if __name__ == "__main__":
    main()
