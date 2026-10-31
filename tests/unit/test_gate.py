"""The quality gate, on the repository's own reports and on fake ones."""

import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from anomaly_inspector.config import load_params
from anomaly_inspector.pipelines import gate

REPO_ROOT = Path(__file__).parents[2]
FLOORS = {"bottle": {"holdout_image_auroc": 0.99, "holdout_aupro": 0.93}}


def write_reports(
    reports: Path, category: str = "bottle", aupro: float = 0.94, p95: float = 80.0
) -> Path:
    folder = reports / category
    folder.mkdir(parents=True)
    metrics = {"holdout_image_auroc": 1.0, "holdout_aupro": aupro}
    (folder / "metrics.json").write_text(json.dumps(metrics))
    (folder / "benchmark.json").write_text(json.dumps({"onnx_fp32": {"p95_ms": p95}}))
    return reports


def failures(checks: list[gate.Check]) -> list[str]:
    return [f"{c.category} {c.name}: {c.detail}" for c in checks if not c.passed]


def test_the_committed_models_pass_the_committed_gates() -> None:
    params = load_params(REPO_ROOT / "params.yaml")
    floors = gate.load_gates(REPO_ROOT / "gates.yaml").floors
    checks = gate.check(
        params.data.categories,
        floors,
        params.promotion.latency_budget_ms,
        REPO_ROOT / "reports/patchcore",
    )
    assert failures(checks) == []
    assert len(checks) == 9  # 3 categories x (2 floors + speed)


def test_a_metric_below_its_floor_fails_with_its_value(tmp_path: Path) -> None:
    checks = gate.check(["bottle"], FLOORS, 200, write_reports(tmp_path, aupro=0.55))
    assert failures(checks) == ["bottle holdout_aupro: 0.550, floor 0.930"]


def test_a_model_slower_than_the_budget_fails(tmp_path: Path) -> None:
    checks = gate.check(["bottle"], FLOORS, 200, write_reports(tmp_path, p95=250))
    assert failures(checks) == ["bottle p95 latency: 250 ms, budget 200 ms"]


def test_missing_floors_reports_and_metrics_fail_instead_of_crashing(tmp_path: Path) -> None:
    write_reports(tmp_path)  # bottle only
    floors = FLOORS | {"screw": {"holdout_aupro": 0.87}, "carpet": {}}
    floors["bottle"] = {"holdout_aupr": 0.93}  # a typo
    checks = gate.check(["bottle", "carpet", "screw", "zipper"], floors, 200, tmp_path)
    assert failures(checks) == [
        f"bottle holdout_aupr: not in {tmp_path / 'bottle/metrics.json'}",
        f"carpet metrics: no {tmp_path / 'carpet/metrics.json'}",
        f"carpet p95 latency: no {tmp_path / 'carpet/benchmark.json'}",
        f"screw metrics: no {tmp_path / 'screw/metrics.json'}",
        f"screw p95 latency: no {tmp_path / 'screw/benchmark.json'}",
        "zipper floors: none in gates.yaml",
        f"zipper metrics: no {tmp_path / 'zipper/metrics.json'}",
        f"zipper p95 latency: no {tmp_path / 'zipper/benchmark.json'}",
    ]


@pytest.mark.parametrize("floors", [{"bottle": {"holdout_aupro": 1.5}}, {"bottle": None}])
def test_impossible_floors_are_rejected(floors: object) -> None:
    with pytest.raises(ValidationError):
        gate.Gates.model_validate({"floors": floors})


@pytest.mark.parametrize(("aupro", "exit_code"), [(0.94, None), (0.55, 1)])
def test_the_command_prints_every_check_and_fails_when_one_does(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    aupro: float,
    exit_code: int | None,
) -> None:
    monkeypatch.chdir(tmp_path)
    params = {"data": {"root": "data/raw/mvtec_ad", "categories": ["bottle"]}}
    Path("params.yaml").write_text(yaml.safe_dump(params))
    Path("gates.yaml").write_text(yaml.safe_dump({"floors": FLOORS}))
    write_reports(Path("reports/patchcore"), aupro=aupro)

    if exit_code is None:
        gate.main()
    else:
        with pytest.raises(SystemExit) as exited:
            gate.main()
        assert exited.value.code == exit_code

    out = capsys.readouterr().out
    assert out.count("\n") == 4  # 3 checks and the summary
    assert ("1 of 3 gates failed" if exit_code else "all 3 gates passed") in out
