# Phase 9: Orchestration with Airflow 3

**Week 7** · [Full details, code and pitfalls](../PROJECT_GUIDE.md#phase-9-orchestration-with-airflow-3)

**Goal:** Two DAGs that run without you. `retrain_patchcore` turns confirmed new images into a challenger and proposes it through a pull request; `batch_inspection` scores a simulated production line every 10 minutes.

## Steps

- [ ] **1.** Give Airflow its own clone of the repo in `~/airflow-workspace/`, with `incoming/`, `line/incoming/`, `review/` and `archive/` next to it, then run `dvc pull`.
- [ ] **2.** Simulate the factory: `scripts/simulate_line.py` copies a mix of pool and defect images into `line/incoming/` every few minutes, and confirmed good ones into `incoming/<category>/good/`.
- [ ] **3.** Start Airflow 3.3.2 from the [official Compose file](https://airflow.apache.org/docs/apache-airflow/stable/howto/docker-compose/index.html): `docker compose up airflow-init`, then `docker compose up -d` (UI on http://localhost:8080).
- [ ] **4.** Edit that Compose file for your laptop:
  - [ ] `LocalExecutor`; delete the redis, airflow-worker and flower services
  - [ ] Mount `/var/run/docker.sock` into the scheduler and add your docker group id with `group_add`
  - [ ] Mount `~/airflow-workspace` at the same absolute path inside Airflow
  - [ ] Join the shared `inspection` network (`external: true`)
  - [ ] Put `INSPECTION_WORKSPACE`, the R2 keys and a fine-grained GitHub token in Airflow's `.env`
- [ ] **5.** Write the retraining DAG, `airflow/dags/retrain_patchcore.py`, and the scripts it calls:
  - [ ] `src/anomaly_inspector/pipelines/ingest.py`: copy confirmed good images from `incoming/` into the training data
  - [ ] `src/anomaly_inspector/pipelines/register.py --all`: register each category's model, call `promote`, print one JSON line with the result per category
  - [ ] `scripts/open_retrain_pr.sh`: create a `retrain/<date>` branch, commit the `.dvc` file and `dvc.lock`, `dvc push`, push the branch and open a pull request
  - [ ] The DAG itself, with the gate, the branch and the `no_change` path
- [ ] **6.** Write the batch inspection DAG, `airflow/dags/batch_inspection.py`: it calls the API over HTTP and scores chunks in parallel with dynamic task mapping.
- [ ] **7.** Test the DAGs with pytest in the official Airflow image (`tests/airflow/test_dags.py`), and add an `airflow-dags` job to `ci.yml`.

## Done when

- [ ] Both DAGs show in the UI with no import errors.
- [ ] 20 confirmed images in `incoming/bottle/good/` lead to a retrain and, if the challenger wins, a pull request that CI checks.
- [ ] A losing challenger ends in `no_change`, and nothing is deployed.
- [ ] batch_inspection runs every 10 minutes; flagged images land in review/ and every result is in the Parquet log.
- [ ] The DAG tests run in CI.

---

[← Phase 8: CI/CD with GitHub Actions](08-ci-cd-github-actions.md) · [All phases](README.md) · [Phase 10: Monitoring and the feedback loop →](10-monitoring-and-feedback-loop.md)
