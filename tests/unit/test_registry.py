"""registry.py: who becomes champion. The rules alone, then the aliases on a file store."""

from pathlib import Path

import mlflow
import pytest
from mlflow import MlflowClient

from anomaly_inspector.config import PromotionConfig
from anomaly_inspector.registry import CHALLENGER, CHAMPION, promote, should_promote

RULES = PromotionConfig(latency_budget_ms=200, aupro_tolerance=0.005)
CHAMP = {"holdout_image_auroc": 0.95, "holdout_aupro": 0.90, "holdout_ms_per_image": 150}


def new(**changes: float) -> dict[str, float]:
    """The champion's metrics with a few changed: the candidate under test."""
    return CHAMP | {f"holdout_{name}": value for name, value in changes.items()}


@pytest.mark.parametrize(
    ("candidate", "promoted"),
    [
        (new(image_auroc=0.97, aupro=0.92), True),  # better at both
        (new(), True),  # as good: the newer one serves
        (new(image_auroc=0.949), False),  # flags parts worse
        (new(aupro=0.896), True),  # locates defects worse, but inside the tolerance
        (new(aupro=0.894), False),  # outside it
        (new(image_auroc=0.99, ms_per_image=201), False),  # better, but too slow for the line
    ],
)
def test_a_clean_candidate_against_the_champion(
    candidate: dict[str, float], promoted: bool
) -> None:
    assert should_promote(candidate, CHAMP, dirty=False, rules=RULES) is promoted


def test_a_dirty_run_never_becomes_champion_however_good() -> None:
    better = new(image_auroc=1.0, aupro=1.0, ms_per_image=10)
    assert not should_promote(better, CHAMP, dirty=True, rules=RULES)
    assert not should_promote(better, None, dirty=True, rules=RULES)


def test_the_first_model_of_a_category_needs_only_to_be_clean_and_fast_enough() -> None:
    assert should_promote(new(image_auroc=0.6), None, dirty=False, rules=RULES)
    assert not should_promote(new(ms_per_image=250), None, dirty=False, rules=RULES)


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> MlflowClient:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    mlflow.set_experiment("patchcore")
    return MlflowClient()


def register(client: MlflowClient, metrics: dict[str, float], dirty: str | None = "False") -> str:
    """A run with these metrics, registered as the next version of patchcore-bottle."""
    with mlflow.start_run() as run:
        mlflow.log_metrics(metrics)
        if dirty is not None:
            mlflow.set_tag("git_dirty", dirty)
    if not client.search_registered_models(filter_string="name = 'patchcore-bottle'"):
        client.create_registered_model("patchcore-bottle")
    source = f"{run.info.artifact_uri}/model"
    return str(client.create_model_version("patchcore-bottle", source, run.info.run_id).version)


def aliases(client: MlflowClient) -> dict[str, str]:
    found = client.get_registered_model("patchcore-bottle").aliases
    return {alias: str(version) for alias, version in found.items()}


def test_the_first_clean_version_becomes_champion(client: MlflowClient) -> None:
    version = register(client, CHAMP)
    assert promote(client, "patchcore-bottle", version, RULES)
    assert aliases(client) == {CHAMPION: "1", CHALLENGER: "1"}


def test_a_better_version_takes_over_and_a_worse_one_stays_challenger(
    client: MlflowClient,
) -> None:
    promote(client, "patchcore-bottle", register(client, CHAMP), RULES)
    assert promote(client, "patchcore-bottle", register(client, new(image_auroc=0.97)), RULES)
    assert aliases(client) == {CHAMPION: "2", CHALLENGER: "2"}

    assert not promote(client, "patchcore-bottle", register(client, new(aupro=0.5)), RULES)
    assert aliases(client) == {CHAMPION: "2", CHALLENGER: "3"}


def test_a_better_version_is_judged_against_the_champion_not_the_last_version(
    client: MlflowClient,
) -> None:
    promote(client, "patchcore-bottle", register(client, CHAMP), RULES)
    promote(client, "patchcore-bottle", register(client, new(image_auroc=0.5)), RULES)  # refused
    # Better than version 2 but worse than the champion (version 1): still refused.
    assert not promote(client, "patchcore-bottle", register(client, new(image_auroc=0.9)), RULES)
    assert aliases(client)[CHAMPION] == "1"


@pytest.mark.parametrize("dirty", ["True", None])
def test_a_dirty_or_unlabelled_run_is_not_promoted(client: MlflowClient, dirty: str | None) -> None:
    version = register(client, CHAMP, dirty=dirty)
    assert not promote(client, "patchcore-bottle", version, RULES)
    assert aliases(client) == {CHALLENGER: "1"}


def test_a_version_without_a_run_is_an_error_not_a_guess(client: MlflowClient) -> None:
    client.create_registered_model("patchcore-bottle")
    version = client.create_model_version("patchcore-bottle", "file:///nowhere/model").version
    with pytest.raises(ValueError, match="no run"):
        promote(client, "patchcore-bottle", str(version), RULES)
