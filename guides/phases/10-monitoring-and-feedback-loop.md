# Phase 10: Monitoring and the feedback loop

**Week 8, days 1 to 3** · [Full details, code and pitfalls](../PROJECT_GUIDE.md#phase-10-monitoring-and-the-feedback-loop)

**Goal:** One Grafana dashboard shows service health and model health, a daily check flags when the line changes, and review decisions flow back into the data.

## Steps

- [ ] **1.** Add custom Prometheus metrics to the API (`anomaly_score_ratio` and `inspection_decisions_total`) and expose `/metrics`.
- [ ] **2.** Add Prometheus (`prom/prometheus:v3.15.1`, scraping `api:8000`) and Grafana (`grafana/grafana:13.2.4`) to Compose under a `monitoring` profile, with the data source and dashboard provisioned from `monitoring/`.
- [ ] **3.** Build the Grafana dashboard and two alert rules (flagged rate and p95 latency) from the queries in the guide.
- [ ] **4.** Save a reference at training time: the evaluate stage writes `reference.parquet` as a DVC output.
- [ ] **5.** Add the daily `drift_report` DAG: it runs in the training image and compares the last 24 hours of the inspection log with the reference (KS test, PSI, flagged rate).
- [ ] **6.** Test the monitoring on purpose: darken the simulated images by 30% for an hour, then write down how long detection took.
- [ ] **7.** Close the loop with human review (a small page served by the API, or Label Studio):
  - [ ] Record each decision (true defect or false alarm) in a reviews table
  - [ ] Confirmed-good false alarms go to `incoming/<category>/good/`
  - [ ] Confirmed defects go to a growing `calib` set; the holdout stays frozen
  - [ ] Add production precision (confirmed defects ÷ flagged parts) to the weekly report

## Done when

- [ ] The provisioned dashboard shows request rate, p95 latency, error rate, the score histogram and the flagged rate.
- [ ] Alert rules exist for the flagged rate and p95 latency.
- [ ] The drift DAG writes a daily report, and the simulated lighting change triggers it.
- [ ] Review decisions are stored, and the weekly report shows production precision.
- [ ] A dashboard screenshot from the drift test is in the README.

---

[← Phase 9: Orchestration with Airflow 3](09-airflow-orchestration.md) · [All phases](README.md) · [Portfolio polish →](11-portfolio-polish.md)
