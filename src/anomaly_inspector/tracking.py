"""One place where every MLflow run starts.

Why one place: PatchCore, STFPM and the experiment scripts will all log runs. If each script
called MLflow by itself, sooner or later one would forget something (the params, the git SHA).
Here we write it once, and every script that calls start_run() records the same things.
"""

import hashlib
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

import mlflow
import yaml

from anomaly_inspector.config import Params

# DVC keeps the images out of git. Git only holds this small "pointer" file about them.
DATA_POINTER = Path("data/raw/mvtec_ad.dvc")
# What `dvc repro` itself rewrites while it runs: the lock file and the reports. They are results,
# not inputs, so a change to them does not stop a run from being rebuilt from its commit.
PIPELINE_OUTPUTS = ("dvc.lock", "reports")


def git(*args: str) -> str:
    # Run "git <args>" in the current folder and return what it prints.
    # check=True: if git fails (not a repo, git not installed), raise an error
    # instead of quietly returning an empty answer.
    result = subprocess.run(["git", *args], capture_output=True, text=True, check=True)
    return result.stdout.strip()


def git_sha() -> str:
    # The 40-character id of the commit we are on. With it, `git checkout <sha>`
    # gives back the exact code of the run.
    return git("rev-parse", "HEAD")


def git_is_dirty() -> bool:
    # True if a file git tracks was changed and not committed. Then the SHA alone does
    # not describe the code that ran, so the run cannot be rebuilt from it.
    # --porcelain prints one line per changed file (nothing when clean).
    # --untracked-files=no ignores new files git does not track (like .vscode/),
    # otherwise every run would look dirty.
    # The :(exclude) paths leave out PIPELINE_OUTPUTS: dvc repro rewrites them during the run,
    # so counting them would mark every pipeline run dirty, and none could be promoted.
    excluded = [f":(exclude){path}" for path in PIPELINE_OUTPUTS]
    return git("status", "--porcelain", "--untracked-files=no", "--", ".", *excluded) != ""


def data_md5() -> str:
    # Inside the pointer file, md5 is a fingerprint of the whole image folder: change,
    # add or remove a single image and the md5 changes. So two runs with the same md5
    # were trained and scored on exactly the same images.
    pointer = yaml.safe_load(DATA_POINTER.read_text())
    return str(pointer["outs"][0]["md5"])


def hardware() -> str:
    # The CPU model. Latency depends on the machine, so two latencies can only be
    # compared when this tag is the same. platform.processor() would be simpler, but
    # it returns an empty string on most Linux machines, so we read the CPU name from
    # /proc/cpuinfo, a file Linux keeps about the processor.
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):  # "model name\t: Intel(R) Core(TM) ..."
                return line.split(":", 1)[1].strip()
    return platform.machine()  # not Linux: at least "x86_64" or "arm64"


def md5_of(path: Path) -> str:
    # A fingerprint of a file's bytes (the same md5 DVC writes in dvc.lock). A run tagged with
    # the md5 of the model it evaluated can be found again from the model file alone.
    digest = hashlib.md5()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tracking_enabled() -> bool:
    # Whether a server is configured. The DVC stages log to MLflow only then, so the same
    # pipeline also runs where there is no server (CI) and keeps its results in files.
    return bool(os.environ.get("MLFLOW_TRACKING_URI", "").strip())


def tracking_uri() -> str:
    # The server's address comes from the environment, never from the code: on your laptop
    # it is http://localhost:5000, inside Docker (Phase 9) it will be http://mlflow:5000.
    # If the variable is missing, MLflow does not complain: it quietly writes the run to a
    # local ./mlflow.db file and a ./mlruns folder, and the run never reaches the server.
    # So we stop with a clear message instead.
    if not tracking_enabled():
        raise RuntimeError(
            "MLFLOW_TRACKING_URI is not set, so MLflow would write this run to ./mlflow.db and "
            "./mlruns instead of the server. Start the server with `make mlflow-up` and run "
            "with `uv run --env-file .env ...`."
        )
    return os.environ["MLFLOW_TRACKING_URI"].strip()


def flatten(params: Params) -> dict[str, Any]:
    # MLflow params are a flat list of name = value. Our params.yaml has sections
    # (data, stfpm), so we join the section and the key with a dot:
    #   {"stfpm": {"lr": 0.4}}  ->  {"stfpm.lr": 0.4}
    # mode="json" turns Python-only types (like Path) into plain text and numbers.
    flat: dict[str, Any] = {}
    for section, values in params.model_dump(mode="json").items():
        for key, value in values.items():
            flat[f"{section}.{key}"] = value
    return flat


def start_run(experiment: str, run_name: str, category: str, params: Params) -> mlflow.ActiveRun:

    # Check the server is configured before anything else, and print where the run goes:
    # if runs ever seem to vanish, this line in the output says where they went.
    tracking_uri()
    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}", flush=True)

    # Tags describe what made the run. We collect them BEFORE the run exists: if git
    # or the DVC file fails here, no half-made run is left behind on the server.
    # Tag values are always text, so the True/False of git_is_dirty becomes "True"/"False".
    # category says which product the model is for (bottle, carpet, screw): each
    # category gets its own model, so it is the first thing we filter on.
    tags = {
        "category": category,
        "git_sha": git_sha(),
        "git_dirty": str(git_is_dirty()),
        "data_md5": data_md5(),
        "hardware": hardware(),
    }

    # An experiment is a folder of runs on the server (for example "patchcore").
    # set_experiment picks that folder, and creates it the first time.
    mlflow.set_experiment(experiment)

    # Ask the server for a new run inside that experiment, with its tags already on it.
    # It starts as RUNNING. We give it back so the caller can write:
    #   with start_run(...) as run:
    # When that `with` block ends, MLflow closes the run by itself:
    # FINISHED if the block went well, FAILED if an error came out of it.
    # log_system_metrics: while the run is open, MLflow samples CPU, RAM, disk and network
    # every 10 seconds (the "System metrics" tab), so a setting too big for the laptop shows.
    run = mlflow.start_run(run_name=run_name, tags=tags, log_system_metrics=True)

    # Params are the inputs of the run: the settings it was trained with. We log the
    # params the script actually uses (passed in), not a fresh read of params.yaml,
    # which could have been edited since. They let us compare and filter runs in the UI.
    mlflow.log_params(flatten(params))
    return run
