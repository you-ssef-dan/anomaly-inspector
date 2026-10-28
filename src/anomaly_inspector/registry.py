"""Champion and challenger: tested code, never a click in the UI, decides which version serves.

Every new version of patchcore-<category> gets the "challenger" alias. It also becomes the
"champion" only if should_promote says so. Anything downstream (the API, Airflow) loads
models:/patchcore-<category>@champion and never a version number.
"""

from collections.abc import Mapping

from mlflow import MlflowClient
from mlflow.entities import Run

from anomaly_inspector.config import PromotionConfig

CHAMPION, CHALLENGER = "champion", "challenger"


def should_promote(
    new: Mapping[str, float],
    old: Mapping[str, float] | None,
    *,
    dirty: bool,
    rules: PromotionConfig,
) -> bool:
    """Whether the new version may replace the champion. old is None for a category's first.

    new and old are the holdout metrics of the two runs, as train_patchcore.py logs them.
    """
    # A run made from uncommitted code cannot be rebuilt from git, so it never serves.
    if dirty:
        return False
    # Too slow for the line, however accurate. The average for now; p95 once it is logged.
    if new["holdout_ms_per_image"] > rules.latency_budget_ms:
        return False
    if old is None:
        return True
    # Never worse at flagging parts, and at most a hair worse at locating defects.
    return (
        new["holdout_image_auroc"] >= old["holdout_image_auroc"]
        and new["holdout_aupro"] >= old["holdout_aupro"] - rules.aupro_tolerance
    )


def run_of(client: MlflowClient, name: str, version: str) -> Run:
    """The run that produced a version: its metrics and tags are what promotion judges."""
    run_id = client.get_model_version(name, version).run_id
    if run_id is None:  # registered by hand, without a run: there is nothing to judge
        raise ValueError(f"{name} version {version} has no run, so no metrics to judge it by")
    return client.get_run(run_id)


def promote(client: MlflowClient, name: str, version: str, rules: PromotionConfig) -> bool:
    """Make the version the challenger, then the champion if it earns it. True if promoted."""
    client.set_registered_model_alias(name, CHALLENGER, version)
    new_run = run_of(client, name, version)

    # Read the aliases instead of asking for the champion and catching the error: the error
    # for "no such alias" is the same MlflowException as for a server that is down.
    aliases = client.get_registered_model(name).aliases
    old = None
    if CHAMPION in aliases:
        old = run_of(client, name, str(aliases[CHAMPION])).data.metrics

    # A missing tag counts as dirty: a run that cannot prove it is clean is not trusted.
    dirty = new_run.data.tags.get("git_dirty") != "False"
    if should_promote(new_run.data.metrics, old, dirty=dirty, rules=rules):
        client.set_registered_model_alias(name, CHAMPION, version)
        return True
    return False
