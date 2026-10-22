# Phase 4: Reproducible pipeline with DVC

**Week 4, days 1 and 2** · [Full details, code and pitfalls](../PROJECT_GUIDE.md#phase-4-reproducible-pipeline-with-dvc)

**Goal:** `dvc repro` rebuilds everything from raw images to a verified ONNX model, re-running only the stages whose inputs changed, and `dvc metrics diff` shows what a change did.

## Steps

- [ ] **1.** Move every tunable value into `params.yaml` and extend the Pydantic `Params` model with one class per section.
- [ ] **2.** Write `dvc.yaml`: `train_patchcore`, `evaluate` and `export` with `foreach` over the categories, plus `train_stfpm` and `evaluate_stfpm`.
- [ ] **3.** Make MLflow logging optional inside the stages: log only when `MLFLOW_TRACKING_URI` is set.
- [ ] **4.** Keep side effects out of `dvc.yaml`: registering, promoting and deploying happen in Airflow or CI.
- [ ] **5.** Run and inspect (with `uv run`): `dvc dag`, `dvc repro`, `dvc exp run -S patchcore.coreset_ratio=0.1`, `dvc exp show`, `dvc metrics diff main --md`, `dvc plots diff main`, `dvc push`.
- [ ] **6.** Run `uv run dvc install` once, so `git push` also pushes DVC outputs.

## Done when

- [ ] `dvc dag` shows prepare, then train, evaluate and export for each category.
- [ ] On a clean clone, `dvc pull` then `dvc status` reports nothing to re-run.
- [ ] Changing `patchcore.coreset_ratio` re-runs only the PatchCore train, evaluate and export stages.
- [ ] `dvc metrics diff main --md` prints a table you can paste into a pull request.
- [ ] `git push` also pushes DVC outputs.

---

[← Phase 3: Experiment tracking and registry with MLflow](03-mlflow-tracking-and-registry.md) · [All phases](README.md) · [Phase 5: ONNX export, optimization and quantization →](05-onnx-export.md)
