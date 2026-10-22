# Phase 1: Data and DVC

**Week 1, days 3 to 5** · [Full details, code and pitfalls](../PROJECT_GUIDE.md#phase-1-data-and-dvc)

**Goal:** The dataset is versioned, validated and split, and anyone can rebuild the processed data with one command.

## Steps

- [ ] **1.** Download bottle, carpet and screw from the [MVTec AD page](https://www.mvtec.com/company/research/datasets/mvtec-ad/downloads) and extract them to `data/raw/mvtec_ad/`.
  - [ ] For Phase 9: move 20% of each category's `train/good` images to `data/simulation/`, so training never sees them.
- [ ] **2.** Run `dvc init` and `dvc add data/raw/mvtec_ad`, then commit the `.dvc` file.
- [ ] **3.** Add a DVC remote: a local folder now (then `dvc push`), Cloudflare R2 later in Phase 8.
- [ ] **4.** Build the manifest in `src/anomaly_inspector/data/manifest.py`: one row per image, paths relative to the repo root.
- [ ] **5.** Validate it with pandera in `src/anomaly_inspector/data/validation.py`, and check that every mask matches its image size and has only the values 0 and 255.
- [ ] **6.** Assign the four splits (`train`, `val_normal`, `calib`, `holdout`), seeded and stratified by defect type.
- [ ] **7.** Wrap steps 4 to 6 in `src/anomaly_inspector/pipelines/prepare.py` and make it the first DVC stage; a second `dvc repro` should do nothing.
- [ ] **8.** Write `src/anomaly_inspector/preprocessing.py` (numpy and Pillow only), shared by training and serving.
- [ ] **9.** Explore the data in `notebooks/01_eda.ipynb` and write `docs/data_card.md`.
- [ ] **10.** Test it all on a tiny fake dataset that a pytest fixture writes to `tmp_path`.

## Done when

- [ ] A second clone in another folder runs `dvc pull` then `dvc repro` and gets an identical manifest.
- [ ] A unit test proves validation fails when a defective image has no mask.
- [ ] reports/data_summary.json has counts per category, split and defect type.
- [ ] A golden-file test pins the output of `preprocess` on a fixture image.
- [ ] docs/data_card.md lists source, license, counts and known issues.

---

[← Phase 0: Repo foundations](00-repo-foundations.md) · [All phases](README.md) · [Phase 2: Models in PyTorch →](02-models-in-pytorch.md)
