# Phase 6: FastAPI inference service and pytest

**Week 5, days 1 to 3** · [Full details, code and pitfalls](../PROJECT_GUIDE.md#phase-6-fastapi-inference-service-and-pytest)

**Goal:** A versioned, validated, tested API that loads the ONNX model once and answers "is this part defective, and where?"

## Steps

- [ ] **1.** Write the inference wrapper (`Inspector`) in `src/anomaly_inspector/serving/inference.py`; it reads the threshold and image size from the model file.
- [ ] **2.** Build the app in `src/anomaly_inspector/serving/app.py` with a factory and a lifespan handler, then add the endpoints (`/metrics` comes in Phase 10):
  - [ ] `GET /health/live`
  - [ ] `GET /health/ready`
  - [ ] `GET /v1/model`
  - [ ] `POST /v1/predict`, with an optional heatmap
  - [ ] `POST /v1/predict/batch`
- [ ] **3.** Test the API with a fake model in `tests/api/`: success, 413, 415, 422 and readiness.
- [ ] **4.** Add the ML tests:
  - [ ] Integration: the real tiny ONNX model behind the real `Inspector` (marked `integration`)
  - [ ] Behaviour (slow): a defective image scores above the threshold, a good one below, and a dark patch raises the score
  - [ ] Property-based (hypothesis): `preprocess` always returns finite values of shape (3, S, S) for modes L, RGB, RGBA and P
  - [ ] Import guard: importing `anomaly_inspector.serving` never loads `torch`
- [ ] **5.** Load-test with Locust, tune uvicorn workers × ONNX threads, and record p95 latency, requests per second and error rate.

## Done when

- [ ] `uv run uvicorn anomaly_inspector.serving.app:app` serves /docs with every endpoint in the table.
- [ ] API tests cover success, 413, 415, 422 and readiness, and coverage stays at or above 80%.
- [ ] The slow behaviour tests pass on the real model.
- [ ] The import guard proves serving never loads torch.
- [ ] Load-test results are in reports/ and summarized in the README.

---

[← Phase 5: ONNX export, optimization and quantization](05-onnx-export.md) · [All phases](README.md) · [Phase 7: Docker and the first deployment →](07-docker-and-first-deployment.md)
