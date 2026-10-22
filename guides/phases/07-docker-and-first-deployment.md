# Phase 7: Docker and the first deployment

**Week 5, days 4 and 5** · [Full details, code and pitfalls](../PROJECT_GUIDE.md#phase-7-docker-and-the-first-deployment)

**Goal:** A small, non-root API image with no PyTorch inside, a Compose stack for local work, and a public demo URL. Reaching this is the MVP gate: start applying.

## Steps

- [ ] **1.** Add a `.dockerignore`.
- [ ] **2.** Write `docker/api.Dockerfile`: multi-stage, non-root user, health check.
- [ ] **3.** Write `docker/train.Dockerfile` with `--extra train` and git (Airflow runs it in Phase 9).
- [ ] **4.** Add the `api` service to `docker-compose.yml` with production-like limits (2 CPUs, 1 GB).
- [ ] **5.** Write `scripts/smoke_test.sh`: health check, one good image and one defective image.
- [ ] **6.** Prove the image has no PyTorch and runs as uid 10001, and note its size for the README.
- [ ] **7.** Add a small demo page at `/`: a static HTML upload form that shows the score, the decision and the heatmap.
- [ ] **8.** Push the image to GHCR and deploy it once by hand (Cloud Run is the recommended host).

## Done when

- [ ] The API image builds, contains no torch, runs as uid 10001, and its size is in the README.
- [ ] `docker compose --profile mlflow --profile api up -d` shows every service healthy.
- [ ] The smoke test passes against the running container.
- [ ] The demo URL works from your phone, and it sits at the top of the README.
- [ ] MVP gate: update your CV and LinkedIn, and start applying while you build weeks 6 to 8.

---

[← Phase 6: FastAPI inference service and pytest](06-fastapi-and-pytest.md) · [All phases](README.md) · [Phase 8: CI/CD with GitHub Actions →](08-ci-cd-github-actions.md)
