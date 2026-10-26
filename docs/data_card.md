# Data card: MVTec AD (bottle, carpet, screw)

This project trains one-class anomaly detectors on three categories of the MVTec Anomaly Detection dataset: a bottle seen from above, a carpet texture and a screw. Models learn from defect-free images only; defective images are used only to choose the decision threshold and to evaluate. The numbers below come from [`notebooks/01_eda.ipynb`](../notebooks/01_eda.ipynb) and [`reports/data_summary.json`](../reports/data_summary.json).

## Source

- **Dataset:** MVTec AD, by MVTec Software GmbH. Official page: <https://www.mvtec.com/research-teaching/datasets/mvtec-ad>
- **Citation:** Paul Bergmann, Michael Fauser, David Sattlegger and Carsten Steger, "A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection", IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2019.
- **How this copy was obtained:** in October 2026 the download links on the official page (hosted on mydrive.ch) returned HTTP 404. The full archive `mvtec_anomaly_detection.tar.xz` (5,264,982,680 bytes) was downloaded from the Hugging Face mirror `micguida1/mvtech_anomaly_detection`. Its SHA-256, as published by Hugging Face, is `cf4313b13603bec67abb49ca959488f7eedce2a9f7795ec54446c649ac98cd3d`, the same value the anomalib library pins for the official archive. The local archive was not kept; the extracted files were checked instead: image counts match the paper, every PNG decodes, and every mask is valid.
- **Extracted:** `bottle/`, `carpet/`, `screw/`, `license.txt` and `readme.txt` into `data/raw/mvtec_ad/`.

## License

**CC BY-NC-SA 4.0** (Attribution, NonCommercial, ShareAlike). The dataset's `readme.txt` adds: "For using the data in a way that falls under the commercial use clause of the license, please contact us."

For this project that means:

- Use for learning and a non-commercial portfolio is allowed, with attribution (the citation above).
- The images are **not** stored in git. DVC keeps them in a private remote.
- A model trained on this data should not be sold or used in a commercial product without MVTec's permission.
- The project's code is Apache-2.0; that license does not extend to the data.

## Version

| What | DVC hash | Files | Size |
| --- | --- | --- | --- |
| `data/raw/mvtec_ad` | `833095ad5b79ed82cfa3de8c579a1ea9.dir` | 1,286 | 940,177,204 bytes |
| `data/simulation` | `315643d242b0e135108ee1972ff6f96e.dir` | 162 | 153,320,560 bytes |

The 1,286 raw files are 1,278 PNGs (1,007 images and 271 masks) plus `license.txt` and `readme.txt` at the top level and inside each category folder.

## Contents

| Category | Image size | Colour | Defect types (test images) |
| --- | --- | --- | --- |
| bottle | 900 × 900 | RGB | broken_large (20), broken_small (22), contamination (21) |
| carpet | 1024 × 1024 | RGB | color (19), cut (17), hole (17), metal_contamination (17), thread (19) |
| screw | 1024 × 1024 | grayscale (`L`) | manipulated_front (24), scratch_head (24), scratch_neck (25), thread_side (23), thread_top (23) |

Every defective image has a pixel mask in `ground_truth/` with the same size as the image and only the values 0 and 255.

## Splits

MVTec provides `train` (good images only) and `test` (good and defective). This project changes them in two steps:

1. **Simulation pool:** 20% of each category's train/good images were moved to `data/simulation/` before anything else (`scripts/make_simulation_pool.py`, seed 42). They stand in for new parts from the line in the Phase 9 factory simulation, and are never used for training or evaluation. The notebook confirms that none of their SHA-1 fingerprints appears in the manifest.
2. **Four splits**, assigned by the `prepare` DVC stage (seed 42): `val_normal` is 10% of the remaining train/good images; the test images are halved into `calib` and `holdout`, stratified by defect type so both halves contain every defect type.

| Category | MVTec train | MVTec test | Pool | train | val_normal | calib (good + defective) | holdout (good + defective) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bottle | 209 | 83 | 42 | 150 | 17 | 41 (10 + 31) | 42 (10 + 32) |
| carpet | 280 | 117 | 56 | 201 | 23 | 58 (14 + 44) | 59 (14 + 45) |
| screw | 320 | 160 | 64 | 230 | 26 | 80 (20 + 60) | 80 (21 + 59) |

**How the splits are used:** `train` is what the models learn from; `val_normal` is for early stopping and a threshold that needs no defects; every decision (threshold, hyperparameters) is made on `calib`; the reported numbers come from `holdout`. Image AUROC on the full test set (`calib` + `holdout`) is also reported, labelled as such, so results can be compared with papers.

## Defect sizes

Share of the image covered by the mask, per defective image:

| Category | Images | Smallest | Median | Largest | Under 1% of the image |
| --- | --- | --- | --- | --- | --- |
| bottle | 63 | 0.58% | 5.72% | 27.42% | 4.8% |
| carpet | 89 | 0.16% | 1.48% | 8.02% | 41.6% |
| screw | 119 | 0.10% | 0.29% | 1.01% | 99.2% |

At the default model resolution of 256 × 256 (65,536 pixels), the median screw defect covers about 189 pixels, roughly 3 cells of PatchCore's 32 × 32 feature grid, and the smallest (a neck scratch) about 65 pixels, a single cell. In carpet, `metal_contamination` (median 0.39%) and `thread` (median 0.64%) are the small ones.

## Processing

- **Validation** (`src/anomaly_inspector/data/validation.py`), run by the `prepare` stage before anything is written: unique paths and SHA-1 fingerprints, only good images in `train`, a mask for every defective image, labels 0 or 1, images at least 64 px wide, and every mask the same size as its image with only the values 0 and 255.
- **Preprocessing** (`src/anomaly_inspector/preprocessing.py`), shared by training and serving: convert to RGB, resize to a square with bilinear interpolation, scale to [0, 1], normalize with the ImageNet mean and standard deviation. Masks must be resized with nearest-neighbour so they stay strictly 0 or 255.

## Known issues and limitations

- **Small evaluation sets.** `holdout` has only 10, 14 and 21 good images for bottle, carpet and screw. One wrongly rejected good part moves the false-reject rate by 10, 7 or 5 points. Results should be read with that uncertainty in mind, ideally with confidence intervals.
- **Small `val_normal`.** With 17, 23 and 26 images, a "99th percentile of good scores" threshold is close to the single highest score, so it is noisy.
- **Tiny defects.** Almost all screw defects and many carpet defects cover under 1% of the image, so the image size is a real trade-off between speed and accuracy. Pixel AUROC is dominated by large defects, so AUPRO is reported as well.
- **Screw orientation.** Good screws lie at any angle, so normal variation is large.
- **Smaller training set than the original.** The simulation pool removed 20% of the official train/good images, so results are not directly comparable to papers that train on the full set. The test set is complete.
- **Controlled capture.** The images come from a fixed setup with controlled lighting and background. Real lines add lighting changes, dust, motion blur and new part variants, which this dataset does not cover.
- **Close to saturated.** State-of-the-art methods already score near-perfect image AUROC on MVTec AD, so small differences between models are within noise. MVTec AD 2 is a harder follow-up.
- **Non-commercial license**, see above.

## Reproduce

```bash
uv sync --all-extras
uv run dvc remote add --local -d localstore <path-to-the-remote>   # or the shared remote from Phase 8
uv run dvc pull
uv run dvc repro      # rebuilds data/processed/manifest.parquet and reports/data_summary.json
```

On a clean clone this produces a byte-identical manifest; a second `dvc repro` does nothing.
