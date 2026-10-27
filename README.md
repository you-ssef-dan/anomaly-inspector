# Anomaly Inspector

[![ci](https://github.com/you-ssef-dan/anomaly-inspector/actions/workflows/ci.yml/badge.svg)](https://github.com/you-ssef-dan/anomaly-inspector/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.12-blue)
![license](https://img.shields.io/badge/code-Apache--2.0-green)

**Visual inspection for factory parts: it learns what a good part looks like, flags the ones that are
not, and draws a heatmap showing where the defect is.** Built end to end on a CPU laptop, from raw
images to a tested, versioned pipeline.

> **Status: work in progress.** Data, models and metrics are done; tracking, the API, Docker,
> CI/CD gates and retraining come next. See the [roadmap](#roadmap).

## The idea

Real factories have very few defective parts, and new kinds of defects keep appearing. So instead of
learning "what a defect looks like", the models learn **what a good part looks like**, from good
images only, and flag anything that differs, including defects they have never seen.

| Model | How it works | Training |
| --- | --- | --- |
| **PatchCore** | Stores the features of small patches of good images. A new patch far from every stored one is suspicious; those distances form the heatmap. | None: one pass over the good images |
| **STFPM** | A student network learns to copy a frozen teacher on good images. Where it copies badly, the image is anomalous. | A real PyTorch training loop with early stopping |

Both are written from the papers in this repo ([`models/`](src/anomaly_inspector/models),
[`training/`](src/anomaly_inspector/training)). PatchCore is also checked against the reference
implementation in [anomalib](https://github.com/open-edge-platform/anomalib).

## Results so far

Dataset: [MVTec AD](https://www.mvtec.com/research-teaching/datasets/mvtec-ad), three categories
(bottle, carpet, screw). The table compares this repo's PatchCore with anomalib's on **bottle**,
with identical inputs, splits, coreset size (1%), seed and metric code. anomalib's backbone is
given the same ImageNet weights, so the only difference is the algorithm code.

| Bottle, full test set (83 images) | Image AUROC | Pixel AUROC | AUPRO |
| --- | --- | --- | --- |
| PatchCore (this repo) | 1.0000 | 0.9866 | 0.9413 |
| anomalib PatchCore | 1.0000 | 0.9868 | 0.9417 |

- **AUPRO** counts every defect region equally, however small, so it is the metric that cares about
  tiny defects. Here it agrees with anomalib's AUPRO to 5 decimals (a slow test checks it).
- Bottle is the easy category. Screw (tiny defects, parts at any angle) is the hard one, see below.
- Full numbers: [`reports/anomalib_benchmark.json`](reports/anomalib_benchmark.json). The data and
  its limits are described in the [data card](docs/data_card.md).

## What changes the result

18 experiments per category, each changing one setting of the PatchCore baseline (WideResNet-50,
256 px, 1% memory bank), plus one combination of two changes. Scored on the `calib` split, so
`holdout` stays untouched.

| Question | Bottle (easy) | Screw (hard) |
| --- | --- | --- |
| **How much does a smaller memory bank cost?** | Almost nothing: 100× smaller, AUPRO 0.944 → 0.934 | A lot: AUPRO 0.916 → 0.865 → **0.546** at 10% / 1% / 0.1% |
| **How many good parts does a new line need?** | One image already gives image AUROC 0.99 | About **50**, at different angles |
| **What does a faster backbone cost?** | AUPRO −0.008, **5× faster** | Nothing measurable, **2.5× faster** |
| **Do small defects need more resolution?** | A little: AUPRO 0.942 → 0.953 | **Yes**: AUPRO 0.865 → **0.941** |

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="reports/figures/experiment_size_dark.png">
  <img alt="Image size against AUPRO, image AUROC and latency" src="reports/figures/experiment_size_light.png">
</picture>

The two cheap wins together, **ResNet-18 at 320 px**, give screw an AUPRO of **0.962** (baseline
0.865) at **38 ms per image** (baseline 169). It was picked after looking at `calib`, so it still has
to be confirmed on `holdout`. All findings and their limits: [`docs/experiments.md`](docs/experiments.md).

## Where it still fails

Screw, the four defects PatchCore comes closest to missing (top) and the four good parts it comes
closest to rejecting (bottom). The white outline is the real defect, the orange wash is the model's
map.

![Screw heatmap gallery](reports/gallery/screw.jpg)

- The missed screw defects are tiny (0.12 to 0.24% of the image) and the map's peak is **not** on
  them, which is why more resolution helps.
- Bottle has no real failure; carpet's misses are thin threads that score like good weave.
- Every panel is explained in [`reports/failure_notes.md`](reports/failure_notes.md), with guesses
  marked as guesses.

## Quick start

You need [uv](https://docs.astral.sh/uv/) (it installs Python 3.12 for you) and `make`.

```bash
git clone https://github.com/you-ssef-dan/anomaly-inspector.git
cd anomaly-inspector
make setup     # installs everything, including the git hooks
make check     # lint, types and tests
```

The tests need **no dataset**: they run on a tiny fake one that the test suite generates. PyTorch is
installed as the CPU build, so there is no GPU or CUDA download.

| Command | What it does |
| --- | --- |
| `make setup` | Install all dependencies and the pre-commit hooks |
| `make check` | Ruff, strict mypy and the fast tests (what CI runs) |
| `make test-slow` | Tests that need the optional `bench` extra, such as the comparison with anomalib |
| `make bench` | Benchmark PatchCore against anomalib on all categories (needs the data and several GB of RAM) |

To rebuild the processed data you need the dataset itself. The [data card](docs/data_card.md)
explains where to get it, how it is verified and how it is split.

## How the project is organised

```
src/anomaly_inspector/
  preprocessing.py   one image function shared by training and (later) serving; numpy + Pillow only
  config.py          typed settings read from params.yaml
  data/              manifest, validation, splits, PyTorch Dataset
  models/            PatchCore, STFPM, backbones
  training/          STFPM training loop (seeded, early stopping, best checkpoint)
  evaluation/        image AUROC, pixel AUROC, AUPRO, and the scoring helpers
  pipelines/         one entry point per DVC stage (prepare so far)
scripts/             benchmark vs anomalib, simulation pool, experiments and charts, gallery
tests/               unit tests on generated fake data, plus a few slow ones
notebooks/           exploratory data analysis (no outputs stored in git)
docs/                data card, experiment findings
reports/             benchmark and experiment results, charts, heatmap gallery, failure notes
params.yaml          every tunable value, validated at load time
dvc.yaml             the data pipeline
```

## Engineering choices

- **No data in git.** Images and models are versioned with [DVC](https://dvc.org); git keeps small
  pointer files. Given the data, a clean clone rebuilds a byte-identical manifest with `dvc repro`.
- **Honest evaluation.** The data is split into `train`, `val_normal`, `calib` and `holdout`.
  Every decision (thresholds, settings) is made on `calib`; the reported numbers come from
  `holdout`. Nothing is tuned on the test set.
- **Same preprocessing everywhere.** One function, with a golden-file test, so training and serving
  cannot drift apart.
- **Reproducible.** Fixed seeds, a locked environment (`uv.lock`) and validated configuration.
- **Checked on every change.** Strict typing, lint, hooks that block large files, notebook outputs
  and secrets, and CI on every pull request. `main` is protected: changes arrive through pull
  requests whose CI must pass.

## Roadmap

✅ done · 🚧 in progress · ⬜ planned

| Phase | What | Status |
| --- | --- | --- |
| 0 | Repo foundations: uv, ruff, mypy, pytest, pre-commit, CI, branch protection | ✅ |
| 1 | Data and DVC: manifest, validation, splits, preprocessing, EDA, data card | ✅ |
| 2 | Models in PyTorch: PatchCore, STFPM, metrics, anomalib benchmark, experiments, heatmap gallery | 🚧 |
| 3 | Experiment tracking and model registry with MLflow | ⬜ |
| 4 | Full DVC pipeline: train, evaluate, export per category | ⬜ |
| 5 | ONNX export, optimisation and quantisation | ⬜ |
| 6 | FastAPI inference service with pytest | ⬜ |
| 7 | Docker and a first deployment | ⬜ |
| 8 | CI/CD with GitHub Actions, including model quality gates | ⬜ |
| 9 | Retraining and batch inspection with Airflow | ⬜ |
| 10 | Monitoring and the feedback loop | ⬜ |

## Data and license

The code is [Apache-2.0](LICENSE). The images are **not** part of this repository: MVTec AD is
licensed [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) (non-commercial), so
models trained on it are for learning and portfolio use. If you use the dataset, please cite:

> Paul Bergmann, Michael Fauser, David Sattlegger and Carsten Steger,
> *A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection*, CVPR 2019.
