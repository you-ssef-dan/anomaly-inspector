"""The quality gate: are the models the pipeline built good enough and fast enough to ship?

  python -m anomaly_inspector.pipelines.gate

Run after `dvc repro` (model-ci runs it on pull requests). For each category in params.yaml it
checks the holdout metrics in reports/patchcore/<category>/metrics.json against the floors in
gates.yaml, and the served model's speed (ONNX FP32, p95 at batch 1, from benchmark.json)
against promotion.latency_budget_ms. It prints every check and exits with 1 if one fails.

ONNX and PyTorch agreement is not checked here: the export stage already fails when they differ.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import yaml
from pydantic import Field

from anomaly_inspector.config import Strict, load_params

REPORTS = Path("reports/patchcore")
GATES_PATH = Path("gates.yaml")

Score = Annotated[float, Field(ge=0, le=1)]


class Gates(Strict):
    floors: dict[str, dict[str, Score]]  # category -> metric in metrics.json -> lowest allowed


def load_gates(path: Path = GATES_PATH) -> Gates:
    return Gates.model_validate(yaml.safe_load(path.read_text()))


@dataclass(frozen=True)
class Check:
    category: str
    name: str
    passed: bool
    detail: str


def read_report(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text()) if path.exists() else None


def check(
    categories: list[str],
    floors: dict[str, dict[str, float]],
    budget_ms: float,
    reports: Path = REPORTS,
) -> list[Check]:
    """One check per floor and one for speed, per category. Missing data fails, never crashes."""
    checks = []
    for category in categories:
        folder = reports / category
        metrics = read_report(folder / "metrics.json")
        benchmark = read_report(folder / "benchmark.json")
        if category not in floors:
            checks.append(Check(category, "floors", False, "none in gates.yaml"))
        if metrics is None:
            checks.append(Check(category, "metrics", False, f"no {folder / 'metrics.json'}"))
        else:
            for name, floor in floors.get(category, {}).items():
                value = metrics.get(name)
                if value is None:
                    detail = f"not in {folder / 'metrics.json'}"
                    checks.append(Check(category, name, False, detail))
                else:
                    detail = f"{value:.3f}, floor {floor:.3f}"
                    checks.append(Check(category, name, value >= floor, detail))
        if benchmark is None:
            checks.append(Check(category, "p95 latency", False, f"no {folder / 'benchmark.json'}"))
        else:
            p95 = benchmark["onnx_fp32"]["p95_ms"]
            checks.append(
                Check(
                    category,
                    "p95 latency",
                    p95 <= budget_ms,
                    f"{p95:.0f} ms, budget {budget_ms:.0f} ms",
                )
            )
    return checks


def main() -> None:
    params = load_params()
    checks = check(params.data.categories, load_gates().floors, params.promotion.latency_budget_ms)
    for c in checks:
        print(f"{'ok' if c.passed else 'FAIL':<4}  {c.category:<8}{c.name:<21}{c.detail}")
    failed = sum(not c.passed for c in checks)
    if failed:
        print(f"{failed} of {len(checks)} gates failed")
        raise SystemExit(1)
    print(f"all {len(checks)} gates passed")


if __name__ == "__main__":
    main()
