# Phase 5: ONNX export, optimization and quantization

**Week 4, days 3 to 5** · [Full details, code and pitfalls](../PROJECT_GUIDE.md#phase-5-onnx-export-optimization-and-quantization)

**Goal:** One self-contained `model.onnx` per category that matches PyTorch, carries its own threshold, and has measured latency in FP32 and INT8.

## Steps

- [ ] **1.** Make the backbone a constructor argument of PatchCore, so tests can use a tiny untrained ResNet-18.
- [ ] **2.** Export to ONNX in `src/anomaly_inspector/export/onnx_export.py`: dynamic batch, one file, example input with batch 2.
- [ ] **3.** Store the threshold, image size, category and git SHA inside the ONNX metadata.
- [ ] **4.** Write the parity tests: raw outputs in `tests/integration/test_onnx_parity.py`, and holdout AUROC for both models in `reports/<category>/export.json` (equal to 3 decimals).
- [ ] **5.** Quantize to INT8 with static calibration on good training images (convolutions only).
- [ ] **6.** Benchmark FP32 and INT8 with warm-up, fixed threads and percentiles, and put the table in the README.
- [ ] **7.** Log the ONNX model to the MLflow registry from now on.

## Done when

- [ ] model.onnx passes `onnx.checker.check_model` and runs in ONNX Runtime at batch 1 and 8.
- [ ] The parity tests pass, and ONNX and PyTorch holdout AUROC match to 3 decimals.
- [ ] The INT8 model is benchmarked, with its accuracy change and speed-up in reports/.
- [ ] Threshold, image size, category and git SHA are inside the ONNX metadata.
- [ ] The registry holds ONNX versions, each category with a champion alias.

---

[← Phase 4: Reproducible pipeline with DVC](04-dvc-pipeline.md) · [All phases](README.md) · [Phase 6: FastAPI inference service and pytest →](06-fastapi-and-pytest.md)
