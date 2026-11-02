"""The dashboards and alert rules ask for metrics that the API really exposes.

A misspelt metric name fails nowhere: the panel shows "No data" and the alert never fires. So
every metric they name is looked up on the API's own /metrics (with a fake model), after a
prediction, a batch and a refused file, so that every series has samples.
"""

import json
import re
from pathlib import Path
from typing import Any

import yaml
from fastapi.testclient import TestClient
from prometheus_client.parser import text_string_to_metric_families

from anomaly_inspector.serving.app import create_app
from anomaly_inspector.serving.settings import Settings
from tests.api.test_app import FakeModel, image_bytes, upload

MONITORING = Path(__file__).parents[2] / "monitoring"
# A metric name right before its label matchers: every query here is written name{...}
SELECTOR = re.compile(r"([a-zA-Z_:][a-zA-Z0-9_:]*)\{")
FROM_PROMETHEUS = {"up", "ALERTS"}  # series Prometheus makes itself, not the API


def panels() -> list[dict[str, Any]]:
    dashboards = sorted((MONITORING / "grafana" / "dashboards").glob("*.json"))
    return [panel for path in dashboards for panel in json.loads(path.read_text())["panels"]]


def queries() -> list[str]:
    rules = yaml.safe_load((MONITORING / "prometheus" / "alerts.yml").read_text())
    alerts = [rule["expr"] for group in rules["groups"] for rule in group["rules"]]
    return [target["expr"] for panel in panels() for target in panel.get("targets", [])] + alerts


def exposed() -> set[str]:
    """Every sample name on /metrics, such as http_requests_total or anomaly_score_ratio_bucket."""
    with TestClient(create_app(Settings(), load=lambda _: FakeModel())) as client:
        client.post("/v1/predict", files={"file": upload("dark.png", image_bytes(20))})
        client.post("/v1/predict/batch", files=[("files", upload("bright.png", image_bytes(240)))])
        client.post("/v1/predict", files={"file": upload("note.txt", b"text", "text/plain")})
        text = client.get("/metrics").text
    families = text_string_to_metric_families(text)
    return {sample.name for family in families for sample in family.samples}


def test_every_metric_the_dashboards_and_alerts_query_is_exposed_by_the_api() -> None:
    named = {name for query in queries() for name in SELECTOR.findall(query)} - FROM_PROMETHEUS
    assert {"inspection_decisions_total", "http_request_duration_seconds_bucket"} <= named
    assert named <= exposed()


def test_every_panel_reads_the_provisioned_data_source() -> None:
    sources = yaml.safe_load(
        (MONITORING / "grafana" / "datasources" / "prometheus.yml").read_text()
    )
    uid = sources["datasources"][0]["uid"]
    with_queries = [panel for panel in panels() if panel["type"] != "row"]
    assert with_queries
    for panel in with_queries:
        uids = {panel["datasource"]["uid"]} | {t["datasource"]["uid"] for t in panel["targets"]}
        assert uids == {uid}, panel["title"]
