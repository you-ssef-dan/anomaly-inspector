# Phase 3: Experiment tracking and registry with MLflow

**Week 3, alongside STFPM** · [Full details, code and pitfalls](../PROJECT_GUIDE.md#phase-3-experiment-tracking-and-registry-with-mlflow)

**Goal:** Every run records its code version, data version, parameters, metrics and artifacts, and the best model per category carries a `champion` alias set by tested code.

## Steps

- [ ] **1.** Run MLflow and PostgreSQL with `docker compose --profile mlflow up -d`; keep `POSTGRES_PASSWORD` and `MLFLOW_TRACKING_URI` in a git-ignored `.env` and commit `.env.example`.
- [ ] **2.** Log every run the same way; the code reads `MLFLOW_TRACKING_URI` from the environment and never hard-codes a server.
- [ ] **3.** Log the same five things in every run:
  - [ ] Params: everything in `params.yaml`, flattened
  - [ ] Tags: git SHA, dirty flag, DVC data hash, category, hardware
  - [ ] Metrics: holdout AUROC, AUPRO, F1, miss rate, false-reject rate, p95 latency, model size; STFPM losses per epoch
  - [ ] Artifacts: failure gallery, ROC and PRO curves, `threshold.json`
  - [ ] Model: PyTorch for now, the ONNX model from Phase 5
- [ ] **4.** Write `should_promote` and `promote`: each new version gets the `challenger` alias, and tested code decides whether it becomes `champion`.
- [ ] **5.** Register one model per category: `patchcore-bottle`, `patchcore-carpet`, `patchcore-screw`.
- [ ] **6.** Turn on `mlflow.enable_system_metrics_logging()`.
- [ ] **7.** Compare the Phase 2 experiments in the MLflow UI and save screenshots for the README.

## Done when

- [ ] `docker compose --profile mlflow up -d` starts MLflow and PostgreSQL, and runs survive a restart.
- [ ] Every run has the git SHA, data hash, all params, holdout metrics and the failure gallery.
- [ ] The coreset and few-shot experiments are compared in the UI, with screenshots saved.
- [ ] Each category has a `champion` alias set by `promote`, and `should_promote` has unit tests.
- [ ] A fresh Python session loads `models:/patchcore-bottle@champion` and reproduces the logged holdout AUROC.

---

[← Phase 2: Models in PyTorch](02-models-in-pytorch.md) · [All phases](README.md) · [Phase 4: Reproducible pipeline with DVC →](04-dvc-pipeline.md)
