# Phase 2: Models in PyTorch

**Weeks 2 and 3** · [Full details, code and pitfalls](../PROJECT_GUIDE.md#phase-2-models-in-pytorch)

**Goal:** Two models you wrote yourself, evaluated with your own tested metrics, and checked against a reference implementation.

## Steps

- [ ] **1.** Write `ManifestDataset`, which reads manifest rows and calls `preprocess`.
- [ ] **2.** Implement PatchCore in `src/anomaly_inspector/models/patchcore.py`: patch features, greedy coreset, nearest-neighbour distance as the score.
- [ ] **3.** Implement STFPM in `src/anomaly_inspector/models/stfpm.py`, with the teacher always in eval mode.
- [ ] **4.** Write the STFPM training loop in `src/anomaly_inspector/training/`: seeded, early stopping on `val_normal` loss, best checkpoint saved.
- [ ] **5.** Implement image AUROC, pixel AUROC and AUPRO (write AUPRO yourself) in `src/anomaly_inspector/evaluation/metrics.py`.
- [ ] **6.** Test the metrics on cases you can work out by hand (perfect, inverted and constant maps), plus one slow test against anomalib's AUPRO.
- [ ] **7.** Benchmark against anomalib's PatchCore, installed as a separate `bench` extra.
- [ ] **8.** Run the four experiments:
  - [ ] Coreset ratio: 0.01 vs 0.1
  - [ ] Number of good training images: 1, 5, 10, 50, all
  - [ ] Backbone: WideResNet-50 vs ResNet-18
  - [ ] Image size: 224, 256, 320
- [ ] **9.** Save a heatmap gallery per category in `reports/` (worst misses and worst false rejects) with two lines on why each one fails.

## Done when

- [ ] PatchCore reaches at least 0.98 image AUROC on bottle. Bottle is easy, so a lower number almost always means a preprocessing bug.
- [ ] The metric tests pass, including the hand-worked cases.
- [ ] STFPM trains with early stopping, and the best checkpoint reloads to the same validation loss.
- [ ] A results table compares your PatchCore, your STFPM and anomalib's PatchCore on all three categories.
- [ ] Running PatchCore twice with the same seed gives identical metrics (a test checks this).
- [ ] The heatmap gallery and failure notes are in reports/.

---

[← Phase 1: Data and DVC](01-data-and-dvc.md) · [All phases](README.md) · [Phase 3: Experiment tracking and registry with MLflow →](03-mlflow-tracking-and-registry.md)
