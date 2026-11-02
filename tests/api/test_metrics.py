"""/metrics, what Prometheus reads: requests and latency per endpoint, and the model's decisions.

Read with Prometheus's own text parser, so the tests see what a scrape sees.
"""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from prometheus_client.parser import text_string_to_metric_families

from anomaly_inspector.serving.app import create_app
from anomaly_inspector.serving.settings import Settings
from tests.api.test_app import FakeModel, image_bytes, upload

DEFECT = 'inspection_decisions_total{category="bottle",result="defect"}'
OK = 'inspection_decisions_total{category="bottle",result="ok"}'


def new_client() -> TestClient:
    return TestClient(create_app(Settings(max_batch=3), load=lambda _: FakeModel()))


@pytest.fixture
def client() -> Iterator[TestClient]:
    with new_client() as test_client:
        yield test_client


def scrape(client: TestClient) -> dict[str, float]:
    """Every sample on /metrics, keyed the way Prometheus shows it: name{label="value",...}."""
    response = client.get("/metrics")
    assert response.status_code == 200
    samples = {}
    for family in text_string_to_metric_families(response.text):
        for sample in family.samples:
            labels = ",".join(f'{key}="{value}"' for key, value in sorted(sample.labels.items()))
            samples[f"{sample.name}{{{labels}}}"] = sample.value
    return samples


def predict(client: TestClient, gray: int) -> None:
    """One photo; the fake model's score is its brightness, gray / 255, against a 0.5 threshold."""
    response = client.post("/v1/predict", files={"file": upload(f"{gray}.png", image_bytes(gray))})
    assert response.status_code == 200


def test_each_decision_is_counted_by_category_and_result(client: TestClient) -> None:
    predict(client, 230)  # defect
    batch = [("files", upload(f"{gray}.png", image_bytes(gray))) for gray in (20, 30, 240)]
    assert client.post("/v1/predict/batch", files=batch).status_code == 200  # ok, ok, defect

    metrics = scrape(client)
    assert (metrics[DEFECT], metrics[OK]) == (2, 2)


def test_the_score_is_recorded_as_a_ratio_of_the_threshold(client: TestClient) -> None:
    predict(client, 191)  # score 0.749, threshold 0.5: ratio 1.498

    metrics = scrape(client)
    assert metrics['anomaly_score_ratio_sum{category="bottle"}'] == pytest.approx(191 / 255 / 0.5)
    assert metrics['anomaly_score_ratio_bucket{category="bottle",le="1.25"}'] == 0
    assert metrics['anomaly_score_ratio_bucket{category="bottle",le="1.5"}'] == 1


def test_requests_are_counted_per_endpoint_and_status(client: TestClient) -> None:
    predict(client, 20)
    text = upload("note.txt", b"not an image", "text/plain")
    assert client.post("/v1/predict", files={"file": text}).status_code == 415

    metrics = scrape(client)
    predict_requests = 'http_requests_total{handler="/v1/predict",method="POST",status="%s"}'
    assert metrics[predict_requests % "2xx"] == 1
    assert metrics[predict_requests % "4xx"] == 1
    assert metrics[OK] == 1  # the refused file was never scored, so it is no decision
    assert DEFECT not in metrics
    # Steps of 50 ms around the 0.1 to 0.2 s this API takes, so a p95 alert can be precise.
    bucket = 'http_request_duration_seconds_bucket{handler="/v1/predict",le="0.15",method="POST"}'
    assert bucket in metrics


def test_health_checks_and_scrapes_are_not_counted(client: TestClient) -> None:
    for path in ("/health/live", "/health/ready", "/metrics", "/metrics"):
        assert client.get(path).status_code == 200

    assert not [key for key in scrape(client) if key.startswith("http_requests_total")]


def test_metrics_are_not_in_the_api_docs(client: TestClient) -> None:
    assert "/metrics" not in client.get("/openapi.json").json()["paths"]


def test_each_app_keeps_its_own_counts() -> None:
    with new_client() as first, new_client() as second:
        predict(first, 20)
        assert scrape(first)[OK] == 1
        assert OK not in scrape(second)
