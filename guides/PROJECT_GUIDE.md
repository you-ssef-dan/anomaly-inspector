# Industrial Anomaly Detection: MLOps Project Guide

Oct 22, 2026 · @youssef

> Local copy of the [live guide](https://claude.ai/artifact/12sJjoLu9n12Ws4c7w4sbq), exported on 22 October 2026. Changes made here do not sync back to the doc.

## Overview

You will build an automated visual inspection system. It learns what a good part looks like from defect-free images, flags defective parts, and draws a heatmap showing where the defect is. Around the model you build the full MLOps lifecycle: versioned data, tracked experiments, a reproducible pipeline, an ONNX model behind a tested FastAPI service in Docker, CI/CD with GitHub Actions, and Airflow retraining.

### Why this project gets interviews

- It is a real factory problem. Defects are rare and change over time, so inspection systems often start with a model trained on good parts only.
- It shows the whole lifecycle hiring managers ask about, not only a notebook.
- It is measurable. MVTec AD is a public benchmark, so you can compare your numbers with published papers.
- It runs on a CPU laptop. That matches your machine (no NVIDIA GPU), free GitHub runners, and many edge PCs on factory lines.

### What makes it advanced

Most portfolio projects skip these six things. Each one is a strong interview story.

1. You implement PatchCore from the paper in PyTorch, then check it against a reference implementation.
2. You evaluate honestly: the decision threshold is set on held-out data, never on the test set, and you report the miss rate and false-reject rate a plant manager cares about.
3. You test training/serving parity: the same preprocessing everywhere, and ONNX outputs checked against PyTorch.
4. You serve with ONNX Runtime and no PyTorch in the image, and you benchmark INT8 quantization.
5. You add quality gates: a pull request that lowers accuracy or slows inference fails CI.
6. You close the loop: Airflow retrains a challenger, compares it with the champion, and promotes it through the MLflow registry.

![Every change reaches production through a pull request and its gates](assets/architecture.svg)

You and Airflow both propose changes as pull requests, and only GitHub Actions, once the tests and model gates pass, builds the image that Cloud Run serves.

### Tech stack and where each tool is used

| Tool | Role in this project | Phase |
| --- | --- | --- |
| uv, ruff, mypy, pre-commit, pytest | Environment, lint, types, tests | 0 |
| DVC | Data and model versioning, reproducible pipeline | 1, 4 |
| PyTorch, torchvision, timm | Feature extraction, PatchCore, STFPM training | 2 |
| MLflow | Experiment tracking, model registry with aliases | 3 |
| ONNX, ONNX Runtime | Portable model, graph optimization, INT8 quantization | 5 |
| FastAPI, Pydantic | Inference API with input validation | 6 |
| Docker, Docker Compose | Training and serving images, local stack | 7 |
| GitHub Actions, GHCR | CI, model quality gates, image publishing, deploy | 8 |
| Airflow 3 | Scheduled retraining and batch inspection | 9 |
| Prometheus, Grafana | Service and model monitoring | 10 |

## Plan

Plan for 8 weeks at 10 to 15 hours a week. Weeks 1 to 5 produce a deployed MVP you can put on your CV and start applying with. Weeks 6 to 8 add the production layer: CI/CD quality gates, Airflow and monitoring.

![A deployed MVP after week 5, the production layer by week 8](assets/roadmap.svg)

Each week ends with a merged pull request; after the week 5 gate you apply for jobs while weeks 6 to 8 add the production layer.

### Scope

MVP, weeks 1 to 5:

- PatchCore on 3 MVTec AD categories: one texture and two objects (for example carpet, bottle, screw).
- STFPM trained from scratch as a fast second model, so you practice a real PyTorch training loop.
- Metrics: image AUROC, pixel AUROC, AUPRO, plus miss rate and false-reject rate at a calibrated threshold.
- Data and pipeline versioned with DVC, every run tracked in MLflow, the best model registered.
- ONNX model with parity tests against PyTorch.
- FastAPI service with pytest tests, in a small Docker image without PyTorch.
- GitHub Actions CI (lint, types, tests, image build) and a live demo link.

Production layer, weeks 6 to 8:

- CI quality gates on model metrics and latency, and automatic deployment.
- Airflow DAGs for retraining (challenger vs champion) and batch inspection.
- Prometheus and Grafana monitoring, a drift check, and the portfolio polish.

Stretch, after week 8 (pick one or two, not all):

- All 15 MVTec AD categories, or one model for all categories.
- A second dataset: VisA, which allows commercial use.
- MVTec AD 2 and its official evaluation server.
- More models benchmarked through anomalib, such as EfficientAD or Dinomaly.
- A DINOv2 backbone for PatchCore, compared with WideResNet-50.
- OpenVINO or TensorRT export, or a small human-review UI that feeds labels back.

### How each phase is written

Every phase follows the same pattern, so you always know when to move on.

- **Goal:** what exists at the end.
- **Steps:** what to do, in order.
- **Code:** only the parts where best practice matters. You write the rest.
- **Pitfalls:** what usually breaks.
- **Done when:** a checklist. Tick it, merge one pull request, then start the next phase.

### Your machine

I checked your laptop on 22 October 2026. The design below is CPU-first because of it.

| Item | What I found | What it means for the project |
| --- | --- | --- |
| GPU | No NVIDIA GPU | PatchCore needs no training, so CPU is fine. Train STFPM at 256 px, or on a free Kaggle or Colab GPU if it is too slow. |
| CPU | 18 threads | Set thread counts explicitly when you benchmark (PyTorch and ONNX Runtime). |
| RAM | 15 GB, with 11 GB and 7 GB of swap in use when checked | Close heavy apps before training. Start only the Compose services you need. |
| Disk | 93 GB free, Btrfs | Enough for datasets, images and caches. Btrfs supports reflinks, so the DVC cache does not duplicate your data. |
| Python | 3.14 as system Python | Let uv install Python 3.12 for the project. Every library in this stack supports it. |
| Docker | Docker 29, you are in the docker group; Podman 5 also installed | Use Docker. Compose and the Airflow quick start assume it. |
| uv | Not installed | Install it first, in Phase 0. |
| Project folder | Empty; the path contains spaces and "&" | Create the repo at a path without spaces, such as \~/projects/industrial-anomaly-detection. Spaces and "&" break Makefiles and shell scripts. |

## Problem framing, datasets and metrics

Treat this as one-class anomaly detection: train on good images only, then score how far each image and each pixel is from normal. Factories work this way because defects are rare, and new defect types appear that no labeled dataset covers.

### Datasets

| Dataset | Size | What is special | License | Role |
| --- | --- | --- | --- | --- |
| [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) | Over 5,000 images, 15 categories, pixel masks | The standard benchmark: textures and objects | CC BY-NC-SA 4.0, no commercial use | Main dataset (MVP) |
| [VisA](https://github.com/amazon-science/spot-diff) | 10,821 images (9,621 normal, 1,200 anomalous), 12 subsets | PCBs, several objects per image, complex structures | CC BY 4.0 | Stretch: second dataset |
| [MVTec AD 2](https://www.mvtec.com/company/research/datasets/mvtec-ad-2) | Over 8,000 images, 8 scenarios | Transparent and overlapping objects, dark-field and back light, tiny defects, lighting changes; a defect-free validation set | CC BY-NC-SA 4.0 | Stretch: harder benchmark with a private test set |

A non-commercial license is fine for a portfolio. Say so in your README and model card; it shows you check licenses, which companies care about.

Start with three MVTec AD categories: bottle (easy, proves your code works), carpet (a texture) and screw (small, rotated parts, hard). Add categories only after the whole pipeline runs end to end.

MVTec AD is close to saturated. The MVTec AD 2 paper reports that state-of-the-art methods stay below 60% average AU-PRO on the new dataset ([arXiv 2503.21622](https://arxiv.org/abs/2503.21622)), which makes it a strong stretch goal.

### Splits: never tune on the test set

MVTec AD gives you train/good plus a test set with good and defective images. There is no validation set with defects, so many tutorials pick the threshold on the test set, which inflates every threshold-based number. Split it like this, with a fixed seed, and save the split as a file that DVC versions.

| Split | Built from | Used for |
| --- | --- | --- |
| train | 90% of train/good | Memory bank (PatchCore), training (STFPM) |
| val\_normal | 10% of train/good | Early stopping, score statistics, a threshold from good parts only |
| calib | 50% of test, stratified by defect type | Every decision: hyperparameters and the threshold |
| holdout | The other 50% of test | The numbers you report |

Also report image AUROC on the full test set, labeled as such, so readers can compare your result with papers.

### Metrics

| Metric | Level | What it tells you |
| --- | --- | --- |
| Image AUROC | Image | How well scores rank defective parts above good ones. No threshold needed. The headline number. |
| Pixel AUROC | Pixel | Localization quality. Large defects and the many normal pixels dominate it. |
| AUPRO, FPR up to 0.3 | Defect region | Every defect region counts the same, so small defects matter. |
| F1 at threshold | Image | Precision and recall at your operating point. |
| Miss rate (FNR) | Image | Defective parts that pass. In a plant, this is the expensive error. |
| False-reject rate (FPR) | Image | Good parts rejected. Costs scrap and manual re-inspection. |
| Latency p50 and p95, images per second | System | Whether the model keeps up with the production line. |

For AUPRO, PRO(t) is the average overlap between the predicted mask at threshold t and each ground-truth defect region C\_k. AUPRO is the area under PRO plotted against the pixel false positive rate, from 0 to 0.3, divided by 0.3.

```latex
\mathrm{PRO}(t)=\frac{1}{K}\sum_{k=1}^{K}\frac{|P_t\cap C_k|}{|C_k|},\qquad \mathrm{AUPRO}=\frac{1}{0.3}\int_{0}^{0.3}\mathrm{PRO}\;d\,\mathrm{FPR}
```

### Choosing the threshold

Implement both methods and compare them on holdout. The gap between them is a strong interview story.

1. **Cost-based, with some defect samples:** on calib, pick the threshold that minimizes 10 × misses + 1 × false rejects. The 10:1 ratio is an assumption; write it in your README.
2. **Good parts only, the cold start of a new product line:** set the threshold at the 99th percentile of val\_normal scores. This aims for about 1% false rejects and needs no defect images.

Save the threshold and score statistics inside the model package. The API reads them from there and never hard-codes them.

## Phase 0: Repo foundations

Week 1, days 1 and 2. **Goal:** a professional skeleton. One command installs everything, one command runs every check, and CI runs the same checks on each pull request.

### Steps

1. Install uv and create the project at a clean path. `uv init` also creates the git repository.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
mkdir -p ~/projects && cd ~/projects
uv init --package --python 3.12 --name anomaly-inspector industrial-anomaly-detection
cd industrial-anomaly-detection
```

2. Create this layout. Each folder is filled by a later phase.

```text
industrial-anomaly-detection/
├── .github/workflows/        # ci.yml now; model-ci.yml and cd.yml in Phase 8
├── airflow/dags/             # Phase 9
├── docker/                   # api.Dockerfile, train.Dockerfile (Phase 7)
├── docs/                     # model card, data card, decision records
├── monitoring/               # Prometheus and Grafana config (Phase 10)
├── notebooks/                # exploration only, outputs stripped
├── src/anomaly_inspector/
│   ├── config.py             # Pydantic models for params.yaml
│   ├── preprocessing.py      # numpy + Pillow only, shared by training and serving
│   ├── data/                 # manifest, validation, splits, Dataset classes
│   ├── models/               # patchcore.py, stfpm.py
│   ├── training/             # STFPM training loop
│   ├── evaluation/           # metrics, thresholds, reports
│   ├── export/               # ONNX export, quantization, parity checks
│   ├── pipelines/            # one entry point per DVC stage
│   └── serving/              # FastAPI app; never imports torch
├── tests/                    # unit/, integration/, api/
└── params.yaml, dvc.yaml, docker-compose.yml, Makefile, pyproject.toml, uv.lock
```

3. Tell uv to use CPU-only PyTorch wheels. Without this, uv installs the CUDA build: gigabytes of NVIDIA libraries your laptop cannot use. Add this to pyproject.toml before adding torch.

```toml
[tool.uv.sources]
torch = [{ index = "pytorch-cpu" }]
torchvision = [{ index = "pytorch-cpu" }]

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true
```

4. Add dependencies in groups, so the serving image never installs PyTorch.

```bash
uv add numpy pillow pydantic pydantic-settings pyyaml
uv add --optional train torch torchvision timm scipy scikit-learn pandas pyarrow \
    matplotlib mlflow "dvc[s3]" onnx onnxscript onnxruntime
uv add --optional serve fastapi "uvicorn[standard]" python-multipart onnxruntime \
    prometheus-fastapi-instrumentator
uv add --dev ruff mypy pytest pytest-cov hypothesis httpx pre-commit
```

5. Configure the tools in pyproject.toml.

```toml
[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "W", "I", "B", "UP", "SIM", "PT", "NPY", "RUF"]

[tool.mypy]
python_version = "3.12"
strict = true

[[tool.mypy.overrides]]
module = ["torchvision.*", "timm.*", "scipy.*", "sklearn.*", "mlflow.*", "onnxruntime.*"]
ignore_missing_imports = true

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra --strict-markers --cov=anomaly_inspector --cov-report=term-missing"
markers = [
  "slow: needs the real dataset or takes more than a few seconds",
  "integration: needs Docker, the network or a real exported model",
]

[tool.coverage.report]
fail_under = 80
```

6. Add pre-commit hooks in `.pre-commit-config.yaml`, then run `uv run pre-commit install`.

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.17.1
    hooks:
      - id: ruff-check
        args: [--fix]
      - id: ruff-format
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v6.0.0
    hooks:
      - id: check-added-large-files
        args: [--maxkb=1000]
      - id: check-yaml
      - id: end-of-file-fixer
      - id: trailing-whitespace
  - repo: https://github.com/kynan/nbstripout
    rev: 0.9.1
    hooks:
      - id: nbstripout
  - repo: https://github.com/gitleaks/gitleaks
    rev: v8.30.1
    hooks:
      - id: gitleaks
```

7. Add a Makefile so every command is one word. Recipe lines must start with a tab.

```makefile
.PHONY: setup lint test check
setup:
	uv sync --all-extras
	uv run pre-commit install
lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy src
test:
	uv run pytest -m "not slow and not integration"
check: lint test
```

8. Add the first CI workflow, `.github/workflows/ci.yml`. Actions are pinned to full commit SHAs, with the version in a comment; Phase 8 explains why.

```yaml
name: ci
on:
  pull_request:
  push:
    branches: [main]
permissions:
  contents: read
concurrency:
  group: ci-${{ github.ref }}
  cancel-in-progress: true
jobs:
  checks:
    runs-on: ubuntu-latest
    timeout-minutes: 20
    steps:
      - uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7.0.1
      - uses: astral-sh/setup-uv@07a047e608db84b3d8c06cd9a7c7dcc9c355c18e # v10.3.2
      - run: uv sync --locked --all-extras
      - run: uv run ruff check .
      - run: uv run ruff format --check .
      - run: uv run mypy src
      - run: uv run pytest -m "not slow and not integration"
```

9. Keep one config file. `params.yaml` holds every tunable value, DVC reads it in Phase 4, and Pydantic validates it so a typo fails loudly instead of silently using a default.

```python
# src/anomaly_inspector/config.py
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DataConfig(Strict):
    root: Path
    categories: list[str]
    image_size: int = Field(256, ge=64)
    val_normal_fraction: float = Field(0.1, gt=0, lt=1)
    seed: int = 42


class PatchCoreConfig(Strict):
    backbone: str = "wide_resnet50_2"
    layers: list[str] = ["layer2", "layer3"]
    coreset_ratio: float = Field(0.1, gt=0, le=1)


class Params(Strict):
    data: DataConfig
    patchcore: PatchCoreConfig


def load_params(path: Path = Path("params.yaml")) -> Params:
    return Params.model_validate(yaml.safe_load(path.read_text()))
```

10. Set up the git workflow. Protect main so merging needs a pull request and a green CI check, even though you work alone. Open one GitHub issue per phase, link each pull request to it, use Conventional Commits (feat:, fix:, test:, ci:, docs:) and squash-merge. Recruiters do read commit history.

### Pitfalls

- Committing data, models or MLflow runs to git. DVC tracks them instead (Phase 1).
- Notebooks with outputs bloating every diff. nbstripout removes them.
- Dependencies that drift. Commit uv.lock; CI runs `uv sync --locked`, which fails if the lock is stale.
- Logic living in notebooks. Notebooks are for exploring; reusable code goes in src/ with tests.

### Done when

- [ ] A fresh clone runs `make setup && make check` with no errors.
- [ ] pre-commit blocks a commit that has a lint error.
- [ ] CI runs on a pull request, passes, and branch protection requires it.
- [ ] `uv run python -c "import torch; print(torch.__version__)"` prints a version ending in +cpu.
- [ ] README has a one-line pitch, the CI badge and setup steps.

## Phase 1: Data and DVC

Week 1, days 3 to 5. **Goal:** the dataset is versioned, validated and split, and anyone can rebuild the processed data with one command.

### Steps

1. Download three categories from the [MVTec AD downloads page](https://www.mvtec.com/company/research/datasets/mvtec-ad/downloads): bottle.tar.xz (148 MB), carpet.tar.xz (705 MB) and screw.tar.xz (186 MB). The full dataset is about 4.9 GB, which you do not need yet. If you plan the factory simulation in Phase 9, move 20% of each category's train/good images to data/simulation/ now, so training never sees them.

```bash
mkdir -p data/raw/mvtec_ad
for c in bottle carpet screw; do tar -xJf ~/Downloads/$c.tar.xz -C data/raw/mvtec_ad; done
```

2. Initialize DVC and track the raw folder. Git versions the small .dvc file that holds the folder's hash; DVC stores the images.

```bash
uv run dvc init
uv run dvc add data/raw/mvtec_ad
git add data/raw/mvtec_ad.dvc data/raw/.gitignore .dvc .dvcignore
git commit -m "data: track MVTec AD bottle, carpet, screw with DVC"
```

3. Add a remote. Start with a folder on your disk; add a cloud remote when CI needs the data in Phase 8. [Cloudflare R2](https://developers.cloudflare.com/r2/pricing/) is S3-compatible, includes 10 GB-month of storage free, and charges nothing for egress, so your 1 GB of images fits.

```bash
# Now: a local remote outside the repo. --local keeps your personal path out of git.
uv run dvc remote add --local -d localstore ~/dvc-storage/industrial-anomaly-detection
uv run dvc push

# Phase 8: Cloudflare R2. Keys go in .dvc/config.local, which git ignores.
uv run dvc remote add -d r2 s3://<bucket>/dvc
uv run dvc remote modify r2 endpointurl https://<account_id>.r2.cloudflarestorage.com
uv run dvc remote modify --local r2 access_key_id <key>
uv run dvc remote modify --local r2 secret_access_key <secret>
uv run dvc remote default --local --unset  # otherwise your local default hides r2
uv run dvc push -r r2
```

4. Build a manifest: one row per image, with paths relative to the repo root so it works in Docker and CI.

```python
# src/anomaly_inspector/data/manifest.py
import hashlib
from pathlib import Path

import pandas as pd
from PIL import Image


def build_manifest(root: Path, categories: list[str]) -> pd.DataFrame:
    rows = []
    for category in categories:
        for path in sorted((root / category).glob("*/*/*.png")):
            source_split, defect_type = path.parts[-3], path.parts[-2]
            if source_split == "ground_truth":
                continue
            mask = root / category / "ground_truth" / defect_type / f"{path.stem}_mask.png"
            with Image.open(path) as im:
                width, height, mode = *im.size, im.mode
            rows.append({
                "path": str(path), "category": category,
                "source_split": source_split, "defect_type": defect_type,
                "label": int(defect_type != "good"),
                "mask_path": str(mask) if mask.exists() else None,
                "width": width, "height": height, "mode": mode,
                "sha1": hashlib.sha1(path.read_bytes()).hexdigest(),
            })
    return pd.DataFrame(rows)
```

5. Validate the manifest with [pandera](https://pandera.readthedocs.io/en/stable/). Also check every mask file: same size as its image, values only 0 and 255.

```python
# src/anomaly_inspector/data/validation.py
import pandera.pandas as pa

manifest_schema = pa.DataFrameSchema(
    {
        "path": pa.Column(str, unique=True),
        "source_split": pa.Column(str, pa.Check.isin(["train", "test"])),
        "label": pa.Column(int, pa.Check.isin([0, 1])),
        "mask_path": pa.Column(str, nullable=True),
        "width": pa.Column(int, pa.Check.ge(64)),
        "sha1": pa.Column(str, unique=True),  # the same image twice = leakage
    },
    checks=[
        pa.Check(lambda df: (df.source_split != "train") | (df.label == 0),
                 error="train must contain only good images"),
        pa.Check(lambda df: (df.label == 0) | df.mask_path.notna(),
                 error="every defective image needs a mask"),
    ],
)
```

6. Assign the four splits from the problem framing section, seeded and stratified by defect type.

```python
from sklearn.model_selection import train_test_split


def assign_splits(df: pd.DataFrame, val_fraction: float, seed: int) -> pd.DataFrame:
    parts = []
    for _, g in df.groupby("category"):
        train, test = g[g.source_split == "train"], g[g.source_split == "test"]
        fit, val = train_test_split(train, test_size=val_fraction, random_state=seed)
        calib, hold = train_test_split(
            test, test_size=0.5, stratify=test.defect_type, random_state=seed
        )
        parts += [fit.assign(split="train"), val.assign(split="val_normal"),
                  calib.assign(split="calib"), hold.assign(split="holdout")]
    return pd.concat(parts, ignore_index=True)
```

7. Wrap steps 4 to 6 in `pipelines/prepare.py` and make it the first DVC stage. Running `uv run dvc repro` twice does nothing the second time, until an input changes.

```yaml
# dvc.yaml
stages:
  prepare:
    cmd: python -m anomaly_inspector.pipelines.prepare
    deps:
      - data/raw/mvtec_ad
      - src/anomaly_inspector/data
      - src/anomaly_inspector/pipelines/prepare.py
    params:
      - data
    outs:
      - data/processed/manifest.parquet
    metrics:
      - reports/data_summary.json:
          cache: false
```

8. Write the shared preprocessing now. Training and serving both call this one function, so they cannot drift apart. It uses only numpy and Pillow, so the API image needs no PyTorch.

```python
# src/anomaly_inspector/preprocessing.py
import numpy as np
from PIL import Image

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(image: Image.Image, size: int) -> np.ndarray:
    """Convert to RGB, resize, normalize. Returns float32 of shape (3, size, size)."""
    rgb = image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    x = np.asarray(rgb, dtype=np.float32) / 255.0
    return ((x - MEAN) / STD).transpose(2, 0, 1)
```

9. Explore in `notebooks/01_eda.ipynb`: a sample grid per defect type, image sizes and modes, and defect area as a percent of the image. If many defects cover under 1% of the image, resolution matters; note that in `docs/data_card.md` with the source, license, counts per split and known issues.
10. Test it all on a tiny fake dataset that a pytest fixture writes to `tmp_path`: a few generated PNGs and one mask. CI never needs the real data for unit tests.

### Pitfalls

- Some MVTec AD categories, such as screw, are grayscale. `preprocess` converts everything to RGB because ImageNet backbones expect 3 channels.
- Resizing masks with bilinear interpolation creates values between 0 and 255. Resize masks with nearest-neighbour.
- Absolute paths in the manifest. They break inside Docker and CI.
- Choosing anything by looking at holdout. Only calib is for decisions.

### Done when

- [ ] A second clone in another folder runs `dvc pull` then `dvc repro` and gets an identical manifest.
- [ ] A unit test proves validation fails when a defective image has no mask.
- [ ] reports/data\_summary.json has counts per category, split and defect type.
- [ ] A golden-file test pins the output of `preprocess` on a fixture image.
- [ ] docs/data\_card.md lists source, license, counts and known issues.

## Phase 2: Models in PyTorch

Weeks 2 and 3. **Goal:** two models you wrote yourself, evaluated with your own tested metrics, and checked against a reference implementation.

|  | PatchCore | STFPM |
| --- | --- | --- |
| Idea | Memory bank of normal patch features; a patch far from all of them is anomalous | A student network learns to copy a frozen teacher on good images; where it fails, the image is anomalous |
| Training | None: one pass of feature extraction | Real training loop with early stopping |
| Backbone | WideResNet-50, layers 2 and 3 | ResNet-18, layers 1 to 3 |
| Why it is here | Strong accuracy; [the paper](https://arxiv.org/abs/2106.08265) reports image AUROC up to 99.6% on MVTec AD | Teaches the PyTorch training loop; small and fast on CPU |

### Steps

1. Write a Dataset that reads manifest rows and calls `preprocess`. This is what guarantees training and serving see identical inputs.

```python
class ManifestDataset(torch.utils.data.Dataset):
    def __init__(self, rows: pd.DataFrame, size: int) -> None:
        self.rows, self.size = rows.reset_index(drop=True), size

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        row = self.rows.iloc[i]
        with Image.open(row.path) as im:
            return torch.from_numpy(preprocess(im, self.size)), int(row.label)
```

2. Implement PatchCore. Three ideas from the paper: locally aware patch features (3×3 average pooling), greedy coreset subsampling, and nearest-neighbour distance as the anomaly score. The distance uses matrix multiplication instead of `torch.cdist`, which keeps the model exportable to ONNX in Phase 5.

```python
# src/anomaly_inspector/models/patchcore.py
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2
from torchvision.models.feature_extraction import create_feature_extractor


class PatchCore(nn.Module):
    def __init__(self, layers: tuple[str, ...] = ("layer2", "layer3"), sigma: float = 4.0) -> None:
        super().__init__()
        backbone = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        self.extractor = create_feature_extractor(backbone, {n: n for n in layers})
        self.extractor.eval().requires_grad_(False)
        self.layers = layers
        self.register_buffer("memory_bank", torch.empty(0, 0))
        self.register_buffer("blur", gaussian_kernel(sigma))

    @torch.no_grad()
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) -> (B, C, h, w) patch features."""
        feats = self.extractor(x)
        maps = [F.avg_pool2d(feats[n], 3, stride=1, padding=1) for n in self.layers]
        size = maps[0].shape[-2:]
        maps = [maps[0]] + [F.interpolate(m, size=size, mode="bilinear") for m in maps[1:]]
        return torch.cat(maps, dim=1)

    @torch.no_grad()
    def fit(self, loader, coreset_ratio: float, seed: int = 0) -> None:
        patches = [self.embed(x).permute(0, 2, 3, 1).flatten(0, 2) for x, _ in loader]
        self.memory_bank = greedy_coreset(torch.cat(patches), coreset_ratio, seed=seed)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns the anomaly map (B, 1, H, W) and the image score (B,)."""
        e = self.embed(x)
        b, c, h, w = e.shape
        dist = nearest_distance(e.permute(0, 2, 3, 1).reshape(-1, c), self.memory_bank)
        patch_scores = dist.reshape(b, 1, h, w)
        amap = F.interpolate(patch_scores, size=x.shape[-2:], mode="bilinear")
        p = self.blur.shape[-1] // 2
        amap = F.conv2d(F.pad(amap, (p, p, p, p), mode="reflect"), self.blur)
        return amap, patch_scores.flatten(1).amax(dim=1)


def nearest_distance(patches: torch.Tensor, bank: torch.Tensor) -> torch.Tensor:
    """Euclidean distance from each patch (N, C) to its nearest bank entry (M, C)."""
    d2 = patches.pow(2).sum(1, keepdim=True) - 2 * patches @ bank.T + bank.pow(2).sum(1)
    return d2.clamp_min(0).amin(dim=1).sqrt()


def greedy_coreset(feats: torch.Tensor, ratio: float, dim: int = 128, seed: int = 0) -> torch.Tensor:
    """k-center greedy selection on a random projection (paper, section 3.2)."""
    g = torch.Generator().manual_seed(seed)
    z = feats @ (torch.randn(feats.shape[1], dim, generator=g) / dim**0.5)
    k = max(1, int(len(feats) * ratio))
    idx = [int(torch.randint(len(feats), (1,), generator=g))]
    min_d = (z - z[idx[0]]).pow(2).sum(1)
    for _ in range(k - 1):
        idx.append(int(min_d.argmax()))
        min_d = torch.minimum(min_d, (z - z[idx[-1]]).pow(2).sum(1))
    return feats[idx]


def gaussian_kernel(sigma: float) -> torch.Tensor:
    r = int(4 * sigma + 0.5)
    t = torch.exp(-torch.arange(-r, r + 1, dtype=torch.float32) ** 2 / (2 * sigma**2))
    t = t / t.sum()
    return (t[:, None] * t[None, :])[None, None]
```

3. Implement STFPM. Override `train()` so the teacher stays in eval mode; otherwise its BatchNorm statistics drift while the student trains.

```python
# src/anomaly_inspector/models/stfpm.py
LAYERS = {"layer1": "layer1", "layer2": "layer2", "layer3": "layer3"}


class STFPM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        teacher = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.teacher = create_feature_extractor(teacher, LAYERS).eval().requires_grad_(False)
        self.student = create_feature_extractor(resnet18(weights=None), LAYERS)

    def train(self, mode: bool = True) -> "STFPM":
        super().train(mode)
        self.teacher.eval()
        return self

    def forward(self, x: torch.Tensor) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        with torch.no_grad():
            t = self.teacher(x)
        return t, self.student(x)


def layer_distance(t: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    return 0.5 * (F.normalize(t, dim=1) - F.normalize(s, dim=1)).pow(2).sum(1, keepdim=True)


def stfpm_loss(t: dict[str, torch.Tensor], s: dict[str, torch.Tensor]) -> torch.Tensor:
    return sum(layer_distance(t[n], s[n]).mean() for n in t)


def stfpm_anomaly_map(t, s, size: tuple[int, int]) -> torch.Tensor:
    maps = [F.interpolate(layer_distance(t[n], s[n]), size=size, mode="bilinear") for n in t]
    return torch.stack(maps).prod(dim=0)  # product across layers, as in the paper
```

4. Write the training loop: seeded, early stopping on val\_normal loss, best checkpoint saved. The STFPM paper (section 4.2) trains with SGD at learning rate 0.4, batch size 32, for 100 epochs on 256×256 images, and keeps the checkpoint with the lowest validation loss. Momentum 0.9 and weight decay 1e-4 below are common defaults, not values from the paper.

```python
def train_stfpm(model: STFPM, train_dl, val_dl, cfg: TrainConfig) -> float:
    torch.manual_seed(cfg.seed)
    opt = torch.optim.SGD(model.student.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=1e-4)
    best, patience = float("inf"), 0
    for epoch in range(cfg.epochs):
        model.train()
        for x, _ in train_dl:
            loss = stfpm_loss(*model(x))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val = sum(stfpm_loss(*model(x)).item() for x, _ in val_dl) / len(val_dl)
        log_metrics({"train_loss": loss.item(), "val_loss": val}, step=epoch)  # MLflow, Phase 3
        if val < best - 1e-4:
            best, patience = val, 0
            torch.save(model.student.state_dict(), cfg.checkpoint_path)
        elif (patience := patience + 1) >= cfg.patience:
            break
    return best
```

5. Implement the metrics in `evaluation/metrics.py`: image AUROC and pixel AUROC with scikit-learn's `roc_auc_score`, and AUPRO yourself. Evaluate every model at 256×256, with masks resized by nearest-neighbour, so comparisons are fair.

```python
import numpy as np
from scipy import ndimage


def aupro(masks: np.ndarray, amaps: np.ndarray, fpr_limit: float = 0.3, steps: int = 200) -> float:
    """masks (N, H, W) bool, amaps (N, H, W) float."""
    regions = []  # (image index, flat pixel indices) of every connected defect region
    for i, m in enumerate(masks):
        labeled, n = ndimage.label(m)
        regions += [(i, np.flatnonzero(labeled == r)) for r in range(1, n + 1)]
    normal = ~masks
    fprs, pros = [], []
    for t in np.linspace(amaps.max(), amaps.min(), steps):  # FPR rises as t falls
        pred = amaps >= t
        fprs.append((pred & normal).sum() / normal.sum())
        pros.append(np.mean([pred[i].ravel()[idx].mean() for i, idx in regions]))
    fprs, pros = np.array(fprs), np.array(pros)
    keep = fprs <= fpr_limit
    x = np.append(fprs[keep], fpr_limit)
    y = np.append(pros[keep], np.interp(fpr_limit, fprs, pros))
    return float(np.trapezoid(y, x) / fpr_limit)
```

6. Test the metrics with cases you can work out by hand: a perfect map gives AUPRO 1.0 and AUROC 1.0, an inverted map gives AUROC 0.0, a constant map gives AUPRO 0.0. Add one slow test that compares your AUPRO with anomalib's on random data.
7. Benchmark against [anomalib](https://github.com/open-edge-platform/anomalib) (Apache-2.0, version 2.7 at the time of writing). Install it as a separate optional extra, `bench`; if uv reports a conflict with your pins, use a separate virtual environment. Then change --all-extras in ci.yml to --extra train --extra serve, so CI never installs it.

```python
from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Patchcore

datamodule = MVTecAD(root="data/raw/mvtec_ad", category="bottle")
model = Patchcore()
engine = Engine()
engine.fit(datamodule=datamodule, model=model)
engine.test(datamodule=datamodule, model=model)
```

8. Run four experiments. Each answers a question an engineering manager would ask, and each becomes a chart in your README.

| Experiment | Values | Question it answers |
| --- | --- | --- |
| Coreset ratio | 0.01, 0.1 | How much accuracy do we lose for a 10× smaller, faster model? |
| Number of good training images | 1, 5, 10, 50, all | How many good parts does a new product line need before go-live? |
| Backbone | WideResNet-50, ResNet-18 | What does a faster backbone cost in accuracy? |
| Image size | 224, 256, 320 | Do small defects need more resolution? |

9. Save a heatmap gallery per category to reports/: the worst misses and the worst false rejects, image plus heatmap overlay. Write two lines on why each one fails. This failure analysis is what interviewers remember.

### Pitfalls

- A frozen backbone left in train mode. Always call `.eval()` and `requires_grad_(False)`.
- Running out of RAM. Patch features for all bottle training images at 256 px are about 1.2 GB. Start with coreset ratio 0.01 on your laptop and extract features in batches.
- Using the newer IMAGENET1K\_V2 weights. The PatchCore paper used the original torchvision WideResNet-50 weights, IMAGENET1K\_V1.
- Comparing models on image AUROC only. Two models can tie on AUROC and differ a lot on AUPRO and latency.

### Done when

- [ ] PatchCore reaches at least 0.98 image AUROC on bottle. Bottle is easy, so a lower number almost always means a preprocessing bug.
- [ ] The metric tests pass, including the hand-worked cases.
- [ ] STFPM trains with early stopping, and the best checkpoint reloads to the same validation loss.
- [ ] A results table compares your PatchCore, your STFPM and anomalib's PatchCore on all three categories.
- [ ] Running PatchCore twice with the same seed gives identical metrics (a test checks this).
- [ ] The heatmap gallery and failure notes are in reports/.

## Phase 3: Experiment tracking and registry with MLflow

Week 3, alongside STFPM. **Goal:** every run records its code version, data version, parameters, metrics and artifacts, and the best model per category carries a `champion` alias set by tested code.

### Steps

1. Run the MLflow server with PostgreSQL in Docker Compose. Artifacts go to a Docker volume through the server. (MinIO's free edition is no longer maintained upstream, so this guide avoids it; use a volume locally, and R2 if you later need S3.)

```dockerfile
# docker/mlflow.Dockerfile
FROM python:3.12-slim
RUN pip install --no-cache-dir "mlflow==3.17.2" "psycopg2-binary>=2.9,<3"
```

```yaml
# docker-compose.yml
services:
  mlflow-db:  # not "postgres": Airflow's Compose file already uses that name
    image: postgres:17
    profiles: [mlflow]
    environment:
      POSTGRES_USER: mlflow
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:?set it in .env}
      POSTGRES_DB: mlflow
    volumes: [pgdata:/var/lib/postgresql/data]
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U mlflow"]
      interval: 5s
      retries: 10

  mlflow:
    build: {context: ., dockerfile: docker/mlflow.Dockerfile}
    profiles: [mlflow]
    command: >
      mlflow server --host 0.0.0.0 --port 5000
      --backend-store-uri postgresql://mlflow:${POSTGRES_PASSWORD}@mlflow-db:5432/mlflow
      --artifacts-destination /mlartifacts
      --allowed-hosts "localhost:*,127.0.0.1:*,mlflow:*"
    ports: ["127.0.0.1:5000:5000"]
    volumes: [mlartifacts:/mlartifacts]
    depends_on:
      mlflow-db: {condition: service_healthy}

networks:
  default:
    name: inspection  # Airflow joins this network in Phase 9

volumes:
  pgdata:
  mlartifacts:
```

Put `POSTGRES_PASSWORD` and `MLFLOW_TRACKING_URI=http://localhost:5000` in a git-ignored `.env`, and commit a `.env.example` without real values. Start it with `docker compose --profile mlflow up -d`.

2. Log every run the same way. MLflow reads `MLFLOW_TRACKING_URI` from the environment, so the code never hard-codes a server.

```python
import mlflow

mlflow.set_experiment("patchcore")
with mlflow.start_run(run_name=f"patchcore-{category}"):
    mlflow.set_tags({
        "category": category,
        "git_sha": git_sha(),
        "git_dirty": str(git_is_dirty()),  # a dirty run cannot be reproduced
        "data_md5": yaml.safe_load(Path("data/raw/mvtec_ad.dvc").read_text())["outs"][0]["md5"],
        "hardware": platform.processor() or platform.machine(),
    })
    mlflow.log_params(flatten(params.model_dump(mode="json")))
    model.fit(train_dl, coreset_ratio=params.patchcore.coreset_ratio, seed=params.data.seed)
    result = evaluate(model, calib_dl, holdout_dl)
    mlflow.log_metrics(result.metrics)
    mlflow.log_dict(result.threshold, "threshold.json")
    mlflow.log_figure(result.failure_gallery, "figures/failures.png")
    info = mlflow.pytorch.log_model(model, name="model", registered_model_name=f"patchcore-{category}")
```

3. Log the same five things in every run.

| What | Examples | Why |
| --- | --- | --- |
| Params | Everything in params.yaml, flattened: `patchcore.coreset_ratio` | Compare runs |
| Tags | git SHA, dirty flag, DVC data hash, category, hardware | Reproduce any run |
| Metrics | Holdout image AUROC, pixel AUROC, AUPRO, F1, miss rate, false-reject rate, p95 latency, model size; STFPM losses per epoch | Compare runs and gate promotions |
| Artifacts | Failure gallery, ROC and PRO curves, threshold.json | Explain the numbers |
| Model | PyTorch now; from Phase 5, only the ONNX model you deploy | Deploy from the registry |

4. Promote with code, never with clicks. Each new version gets the `challenger` alias. A pure, unit-tested function decides whether it replaces the `champion`.

```python
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException


def should_promote(new: dict[str, float], old: dict[str, float], budget_ms: float) -> bool:
    return (
        new["holdout_image_auroc"] >= old["holdout_image_auroc"]
        and new["holdout_aupro"] >= old["holdout_aupro"] - 0.005
        and new["latency_p95_ms"] <= budget_ms
    )


def promote(client: MlflowClient, name: str, version: str, budget_ms: float) -> bool:
    client.set_registered_model_alias(name, "challenger", version)
    new = client.get_run(client.get_model_version(name, version).run_id).data.metrics
    try:
        champion = client.get_model_version_by_alias(name, "champion")
    except MlflowException:  # first model for this category
        client.set_registered_model_alias(name, "champion", version)
        return True
    old = client.get_run(champion.run_id).data.metrics
    if should_promote(new, old, budget_ms):
        client.set_registered_model_alias(name, "champion", version)
        return True
    return False
```

5. Register one model per category (`patchcore-bottle`, `patchcore-carpet`, `patchcore-screw`), because each PatchCore memory bank belongs to one product. Anything downstream loads `models:/patchcore-bottle@champion`, never a version number.
6. Turn on `mlflow.enable_system_metrics_logging()` to record CPU and RAM use per run. On a 15 GB laptop, this shows which settings will not fit.
7. Compare the Phase 2 experiments in the MLflow UI with the parallel coordinates and scatter views. Save screenshots for the README.

### Pitfalls

- "Invalid Host header" when another container calls `http://mlflow:5000`. Since MLflow 3.5 the server accepts only localhost and private IPs by default, so the Compose service name must be in `--allowed-hosts` ([MLflow network security](https://mlflow.org/docs/latest/self-hosting/security/network/)).
- Runs going to a local file instead of your server because `MLFLOW_TRACKING_URI` was not set. Print the tracking URI at the start of every run.
- Reloading a PatchCore checkpoint fails because `memory_bank` starts as shape (0, 0). Set `model.memory_bank = torch.empty_like(state["memory_bank"])` before `load_state_dict`.
- Comparing latency measured on different machines. Gate only on latency measured on the same hardware; the `hardware` tag tells you which.

### Done when

- [ ] `docker compose --profile mlflow up -d` starts MLflow and PostgreSQL, and runs survive a restart.
- [ ] Every run has the git SHA, data hash, all params, holdout metrics and the failure gallery.
- [ ] The coreset and few-shot experiments are compared in the UI, with screenshots saved.
- [ ] Each category has a `champion` alias set by `promote`, and `should_promote` has unit tests.
- [ ] A fresh Python session loads `models:/patchcore-bottle@champion` and reproduces the logged holdout AUROC.

## Phase 4: Reproducible pipeline with DVC

Week 4, days 1 and 2. **Goal:** `dvc repro` rebuilds everything from raw images to a verified ONNX model, re-running only the stages whose inputs changed, and `dvc metrics diff` shows what a change did.

DVC is now maintained by lakeFS, which took it over from Iterative in November 2025; it stays open source under the same license, with no breaking changes ([DVC blog](https://dvc.org/blog/dvc-joins-lakefs-your-questions-answered/)).

Interviewers often ask why you use DVC, MLflow and Airflow together. Each owns a different job:

| Tool | Owns | Does not own |
| --- | --- | --- |
| DVC | The graph from data to model, caching, data and model versions, metric diffs between commits | Scheduling, retries, alerts |
| MLflow | Run history, comparisons in the UI, the registry and its aliases | File versioning, the order of steps |
| Airflow | When things run, retries, steps across systems (promote, deploy, notify) | The ML logic itself |

### Steps

1. Put every tunable value in `params.yaml`, and extend the Pydantic `Params` model from Phase 0 with one class per section.

```yaml
data:
  root: data/raw/mvtec_ad
  categories: [bottle, carpet, screw]
  image_size: 256
  val_normal_fraction: 0.1
  seed: 42
patchcore:
  backbone: wide_resnet50_2
  layers: [layer2, layer3]
  coreset_ratio: 0.01
stfpm:
  epochs: 100
  lr: 0.4
  batch_size: 32
  patience: 10
threshold:
  miss_cost: 10
  false_reject_cost: 1
  normal_quantile: 0.99
export:
  quantize: true
```

2. Write the pipeline. `foreach` creates one stage per category (`train_patchcore@bottle` and so on), and `${data.categories}` comes from params.yaml. Add `train_stfpm` and `evaluate_stfpm` the same way, with the stfpm params.

```yaml
# dvc.yaml (the prepare stage from Phase 1 stays first)
stages:
  train_patchcore:
    foreach: ${data.categories}
    do:
      cmd: python -m anomaly_inspector.pipelines.train_patchcore --category ${item}
      deps:
        - data/processed/manifest.parquet
        - src/anomaly_inspector/models
        - src/anomaly_inspector/pipelines/train_patchcore.py
      params:
        - data.image_size
        - patchcore
      outs:
        - models/patchcore/${item}/weights.pt

  evaluate:
    foreach: ${data.categories}
    do:
      cmd: python -m anomaly_inspector.pipelines.evaluate --category ${item}
      deps:
        - models/patchcore/${item}/weights.pt
        - data/processed/manifest.parquet
        - src/anomaly_inspector/evaluation
      params:
        - threshold
      outs:
        - models/patchcore/${item}/threshold.json
        - reports/${item}/figures
      metrics:
        - reports/${item}/metrics.json:
            cache: false
      plots:
        - reports/${item}/roc.csv:
            cache: false
            x: fpr
            y: tpr

  export:
    foreach: ${data.categories}
    do:
      cmd: python -m anomaly_inspector.pipelines.export --category ${item}
      deps:
        - models/patchcore/${item}/weights.pt
        - models/patchcore/${item}/threshold.json
        - src/anomaly_inspector/export
      params:
        - export
      outs:
        - models/patchcore/${item}/model.onnx
      metrics:
        - reports/${item}/export.json:
            cache: false
```

3. Make MLflow logging optional inside the stages: log when `MLFLOW_TRACKING_URI` is set, skip it otherwise. CI then runs the same pipeline with no server, and reads metrics from the DVC metrics files.
4. Keep side effects out of dvc.yaml. Stages turn files into files. Registering, promoting and deploying happen in Airflow or CI, so re-running a stage never creates a duplicate registry version.
5. Run, inspect and compare.

```bash
uv run dvc dag                                     # print the graph
uv run dvc repro                                   # run what changed
uv run dvc exp run -S patchcore.coreset_ratio=0.1  # try a value without editing files
uv run dvc exp show                                # table of experiments
uv run dvc metrics diff main --md                  # metric changes vs main, as Markdown
uv run dvc plots diff main                         # ROC curves, before and after
uv run dvc push                                    # share outputs with CI and Airflow
```

6. Run `uv run dvc install` once. It adds git hooks, so `git push` also pushes DVC outputs and `git checkout` restores the matching data.

### Pitfalls

- A stage reads a file that is not in its `deps`. DVC then skips it when that file changes, and results go stale silently.
- `deps` that are too wide, like all of src/. Then a docstring edit retrains every model. List the narrowest folders.
- Writing results outside `outs` or `metrics`. They are not versioned, so nobody can reproduce them.
- Forgetting `dvc push`. CI and Airflow then fail to find the models. The hooks from step 6 prevent it.

### Done when

- [ ] `dvc dag` shows prepare, then train, evaluate and export for each category.
- [ ] On a clean clone, `dvc pull` then `dvc status` reports nothing to re-run.
- [ ] Changing `patchcore.coreset_ratio` re-runs only the PatchCore train, evaluate and export stages.
- [ ] `dvc metrics diff main --md` prints a table you can paste into a pull request.
- [ ] `git push` also pushes DVC outputs.

## Phase 5: ONNX export, optimization and quantization

Week 4, days 3 to 5. **Goal:** one self-contained `model.onnx` per category that matches PyTorch, carries its own threshold, and has measured latency in FP32 and INT8.

Since PyTorch 2.9, `torch.onnx.export` uses the `torch.export`-based exporter by default (`dynamo=True`), and `dynamic_shapes` replaces the old `dynamic_axes` ([PyTorch 2.14 docs](https://docs.pytorch.org/docs/2.14/onnx_export.html)). It needs the `onnxscript` package, which you added in Phase 0.

### Steps

1. Make the backbone a constructor argument of PatchCore. Tests can then build a tiny model from an untrained ResNet-18, with no weight download, and export it in seconds.
2. Export with a dynamic batch, as one file.

```python
# src/anomaly_inspector/export/onnx_export.py
import torch


def export_onnx(model: torch.nn.Module, path: str, size: int) -> None:
    model.eval()
    example = torch.randn(2, 3, size, size)  # batch 2, not 1: see Pitfalls
    torch.onnx.export(
        model,
        (example,),
        path,
        input_names=["image"],
        output_names=["anomaly_map", "score"],
        dynamic_shapes={"x": {0: "batch"}},  # key = the forward() argument name
        external_data=False,  # single file; fine under 2 GB
    )
```

3. Store what serving needs inside the model file, so the API never hard-codes it.

```python
import onnx

m = onnx.load(path)
onnx.checker.check_model(m)
onnx.helper.set_model_props(m, {
    "category": category, "image_size": "256",
    "threshold": str(threshold), "git_sha": git_sha(),
})
onnx.save(m, path)
# the API reads it back with session.get_modelmeta().custom_metadata_map
```

4. Test parity at two levels: raw outputs, and the metric that matters.

```python
# tests/integration/test_onnx_parity.py
import numpy as np
import onnxruntime as ort
import pytest
import torch


@pytest.mark.parametrize("batch", [1, 4])
def test_onnx_matches_pytorch(tiny_patchcore, tmp_path, batch):
    path = tmp_path / "model.onnx"
    export_onnx(tiny_patchcore, str(path), size=128)
    x = torch.randn(batch, 3, 128, 128)
    amap_pt, score_pt = tiny_patchcore(x)
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    amap_ort, score_ort = sess.run(None, {"image": x.numpy()})
    np.testing.assert_allclose(score_ort, score_pt.numpy(), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(amap_ort, amap_pt.numpy(), rtol=1e-3, atol=1e-3)
```

In the export stage, also score holdout with ONNX Runtime and write both AUROCs to `reports/<category>/export.json`. They must match to 3 decimals.

5. Quantize to INT8 with static calibration on good training images. Quantize only the convolutions; the nearest-neighbour distances stay in FP32 because small distance errors flip decisions.

```python
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process


class GoodImages(CalibrationDataReader):
    def __init__(self, paths: list[str], size: int) -> None:
        self._it = iter([{"image": preprocess(Image.open(p), size)[None]} for p in paths])

    def get_next(self) -> dict | None:
        return next(self._it, None)


quant_pre_process("model.onnx", "model.prep.onnx")
quantize_static(
    "model.prep.onnx",
    "model.int8.onnx",
    calibration_data_reader=GoodImages(train_paths[:100], 256),
    quant_format=QuantFormat.QDQ,
    per_channel=True,
    weight_type=QuantType.QInt8,
    activation_type=QuantType.QUInt8,
    op_types_to_quantize=["Conv"],
)
```

6. Benchmark properly: warm-up runs, fixed thread count, percentiles, hardware recorded.

```python
import time

import numpy as np
import onnxruntime as ort


def benchmark(path: str, batch: int, threads: int, runs: int = 50, warmup: int = 10) -> dict:
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    x = np.random.rand(batch, 3, 256, 256).astype(np.float32)
    for _ in range(warmup):
        sess.run(None, {"image": x})
    ms = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, {"image": x})
        ms.append((time.perf_counter() - t0) * 1000)
    return {"p50_ms": float(np.percentile(ms, 50)), "p95_ms": float(np.percentile(ms, 95)),
            "images_per_s": batch * 1000 / float(np.mean(ms))}
```

Report one table in the README. Rows: PyTorch FP32, ONNX Runtime FP32, ONNX Runtime INT8. Columns: p50 and p95 latency at batch 1, images per second at batch 8, file size, holdout image AUROC and AUPRO. Name the CPU and thread count under it.

7. Register the ONNX model from now on, so the registry holds exactly what you deploy.

```python
mlflow.onnx.log_model(
    onnx.load(path), name="model", registered_model_name=f"patchcore-{category}",
    save_as_external_data=False, metadata={"threshold": threshold, "image_size": 256},
)
```

### Pitfalls

- Exporting with a batch of 1. `torch.export` treats size-1 dimensions as constants, so the dynamic batch breaks. Use a batch of 2 for the example input.
- A second file next to model.onnx. The exporter writes weights as external data by default; `external_data=False` gives one file, or you must ship both.
- Calibrating quantization on calib or holdout images. That leaks test data into the model. Use training images.
- Tolerances that are too tight. Float math reorders between frameworks, so 1e-7 fails for no real reason. Use 1e-4 and 1e-3, and compare metrics too.
- Benchmarks without warm-up, or with default threads. The first runs include memory allocation, and thread defaults differ between libraries.

### Done when

- [ ] model.onnx passes `onnx.checker.check_model` and runs in ONNX Runtime at batch 1 and 8.
- [ ] The parity tests pass, and ONNX and PyTorch holdout AUROC match to 3 decimals.
- [ ] The INT8 model is benchmarked, with its accuracy change and speed-up in reports/.
- [ ] Threshold, image size, category and git SHA are inside the ONNX metadata.
- [ ] The registry holds ONNX versions, each category with a champion alias.

## Phase 6: FastAPI inference service and pytest

Week 5, days 1 to 3. **Goal:** a versioned, validated, tested API that loads the ONNX model once and answers "is this part defective, and where?"

| Method | Path | Purpose |
| --- | --- | --- |
| GET | /health/live | The process is up (liveness) |
| GET | /health/ready | The model is loaded (readiness) |
| GET | /v1/model | Category, threshold, image size, model version |
| POST | /v1/predict | One image in; score, decision and optional heatmap out |
| POST | /v1/predict/batch | Several images in one request |
| GET | /metrics | Prometheus metrics (Phase 10) |

### Steps

1. Write an inference wrapper that reads everything from the model file, including the threshold stored in Phase 5.

```python
# src/anomaly_inspector/serving/inference.py  (numpy, Pillow, onnxruntime only)
from dataclasses import dataclass

import numpy as np
import onnxruntime as ort
from PIL import Image

from anomaly_inspector.preprocessing import preprocess


@dataclass(frozen=True)
class Prediction:
    score: float
    is_anomalous: bool
    anomaly_map: np.ndarray


class Inspector:
    def __init__(self, model_path: str, threads: int) -> None:
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        self.session = ort.InferenceSession(model_path, so, providers=["CPUExecutionProvider"])
        meta = self.session.get_modelmeta().custom_metadata_map
        self.threshold = float(meta["threshold"])
        self.image_size = int(meta["image_size"])
        self.category = meta["category"]
        self.version = meta["git_sha"]

    def predict(self, image: Image.Image) -> Prediction:
        x = preprocess(image, self.image_size)[None]
        amap, score = self.session.run(None, {"image": x})
        s = float(score[0])
        return Prediction(score=s, is_anomalous=s >= self.threshold, anomaly_map=amap[0, 0])
```

2. Build the app with a factory and a lifespan handler. The factory lets tests inject a fake model; the lifespan loads the real one once at startup, and the service fails fast if the file is missing.

```python
# src/anomaly_inspector/serving/app.py
import io
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request, UploadFile
from PIL import Image, UnidentifiedImageError

from .inference import Inspector
from .schemas import PredictResponse
from .settings import Settings

ALLOWED_TYPES = {"image/png", "image/jpeg"}


def default_loader() -> Inspector:
    s = Settings()  # pydantic-settings: API_MODEL_PATH, API_THREADS, API_MAX_UPLOAD_MB
    return Inspector(str(s.model_path), s.threads)


def get_inspector(request: Request) -> Inspector:
    return request.app.state.inspector


def create_app(load: Callable[[], Inspector] = default_loader) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.inspector = load()
        yield

    app = FastAPI(title="Anomaly Inspector", version="1.0.0", lifespan=lifespan)

    @app.get("/health/ready")
    def ready(inspector: Annotated[Inspector, Depends(get_inspector)]) -> dict[str, str]:
        return {"status": "ready", "model_version": inspector.version}

    @app.post("/v1/predict", response_model=PredictResponse)
    def predict(  # plain def: FastAPI runs it in a thread pool, so the event loop stays free
        file: UploadFile,
        inspector: Annotated[Inspector, Depends(get_inspector)],
        heatmap: bool = False,
    ) -> PredictResponse:
        if file.content_type not in ALLOWED_TYPES:
            raise HTTPException(415, "Upload a PNG or JPEG image")
        max_bytes = Settings().max_upload_mb * 1024 * 1024
        data = file.file.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise HTTPException(413, "Image is too large")
        try:
            image = Image.open(io.BytesIO(data))
            image.load()
        except (UnidentifiedImageError, OSError) as err:
            raise HTTPException(422, "Could not decode the image") from err
        t0 = time.perf_counter()
        pred = inspector.predict(image)
        return PredictResponse(
            category=inspector.category, anomaly_score=pred.score,
            threshold=inspector.threshold, is_anomalous=pred.is_anomalous,
            model_version=inspector.version,
            latency_ms=(time.perf_counter() - t0) * 1000,
            heatmap_png_base64=render_heatmap(image, pred) if heatmap else None,
        )

    return app


app = create_app()  # uvicorn anomaly_inspector.serving.app:app
```

Read `Settings` once in a real app (store it on `app.state`); it is inline here to keep the example short. Render the heatmap overlay with numpy and Pillow only, so the image needs neither OpenCV nor matplotlib.

3. Test the API with a fake model. These tests run in milliseconds and need no data.

```python
# tests/api/test_predict.py
import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from anomaly_inspector.serving.app import create_app
from anomaly_inspector.serving.inference import Prediction


class FakeInspector:
    threshold, image_size, category, version = 0.5, 64, "bottle", "test-sha"

    def predict(self, image: Image.Image) -> Prediction:
        return Prediction(score=0.9, is_anomalous=True, anomaly_map=np.zeros((64, 64), np.float32))


@pytest.fixture
def client():
    with TestClient(create_app(load=FakeInspector)) as c:
        yield c


def png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), "gray").save(buf, format="PNG")
    return buf.getvalue()


def test_predict_returns_decision(client):
    r = client.post("/v1/predict", files={"file": ("part.png", png(), "image/png")})
    assert r.status_code == 200
    assert r.json()["is_anomalous"] is True
    assert r.json()["threshold"] == 0.5


@pytest.mark.parametrize(("name", "content", "mime", "status"), [
    ("notes.txt", b"hello", "text/plain", 415),
    ("broken.png", b"\x89PNG not really", "image/png", 422),
])
def test_rejects_bad_uploads(client, name, content, mime, status):
    r = client.post("/v1/predict", files={"file": (name, content, mime)})
    assert r.status_code == status
```

4. Add the tests that make this an ML service, not just a web service:

- **Integration:** the real tiny ONNX model from Phase 5 behind the real `Inspector`, marked `integration`.
- **Behaviour (slow):** a real defective image scores above the threshold, a good one below. Then paste a small dark patch onto a good image; its score must go up.
- **Property-based:** with hypothesis, random sizes and modes (L, RGB, RGBA, P) always give `preprocess` an output of shape (3, S, S) with finite values.
- **Import guard:** a test that imports `anomaly_inspector.serving` and asserts `"torch" not in sys.modules`.

5. Load-test with Locust (add `locust` to the dev group), then tune.

```bash
uv run locust -f tests/load/locustfile.py --host http://localhost:8000 \
    --headless -u 10 -r 2 -t 2m --csv reports/load
```

Keep uvicorn workers × ONNX Runtime threads at or below the CPU cores you give the container. Record p95 latency, requests per second and the error rate for the setting you choose.

### Pitfalls

- `async def` endpoints that run inference. They block the event loop, so one slow request stalls all others. Use plain `def`.
- Loading the model per request. Load it once in the lifespan handler.
- Reading the whole upload before checking its size. Read at most the limit plus one byte.
- Trusting the content type alone. Always decode with Pillow and turn decode errors into 422.

### Done when

- [ ] `uv run uvicorn anomaly_inspector.serving.app:app` serves /docs with every endpoint in the table.
- [ ] API tests cover success, 413, 415, 422 and readiness, and coverage stays at or above 80%.
- [ ] The slow behaviour tests pass on the real model.
- [ ] The import guard proves serving never loads torch.
- [ ] Load-test results are in reports/ and summarized in the README.

## Phase 7: Docker and the first deployment

Week 5, days 4 and 5. **Goal:** a small, non-root API image with no PyTorch inside, a Compose stack for local work, and a public demo URL. Reaching this is the MVP gate: start applying.

### Steps

1. Add a `.dockerignore` so the build context stays small and secrets stay out.

```text
.git
.venv
.env
data/
.dvc/cache
mlruns/
mlartifacts/
notebooks/
reports/
**/__pycache__
```

2. Write the API image as a multi-stage build. Dependencies install in their own cached layer, the project installs as a normal (non-editable) package, and only the virtual environment and the model reach the final image.

```dockerfile
# docker/api.Dockerfile
FROM python:3.12-slim-trixie AS builder
COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-dev --extra serve --no-install-project
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --extra serve --no-editable

FROM python:3.12-slim-trixie
ARG MODEL_FILE=models/patchcore/bottle/model.onnx
RUN useradd --create-home --uid 10001 app
WORKDIR /app
COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --chown=app:app ${MODEL_FILE} /app/model/model.onnx
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1 API_MODEL_PATH=/app/model/model.onnx
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s CMD \
    python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/ready')"
CMD ["uvicorn", "anomaly_inspector.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

3. Write the training image the same way, with `--extra train`, plus git (DVC needs it). Install dependencies only (--no-install-project) and set PYTHONPATH=/repo/src, so the code always comes from the mounted clone at its exact commit, never from a stale image. It installs CPU PyTorch through the index you set in Phase 0. Airflow runs this image in Phase 9.
4. Add the API to Compose, with resource limits close to production so local latency numbers mean something.

```yaml
# docker-compose.yml, next to mlflow-db and mlflow
  api:
    build:
      context: .
      dockerfile: docker/api.Dockerfile
      args: {MODEL_FILE: models/patchcore/bottle/model.onnx}
    profiles: [api]
    ports: ["127.0.0.1:8000:8000"]
    environment: {API_THREADS: "2"}
    deploy:
      resources:
        limits: {cpus: "2", memory: 1g}
```

```bash
docker compose --profile mlflow --profile api up -d --build
docker compose ps                      # both services should show (healthy)
docker image ls                        # write the API image size in the README
```

5. Write `scripts/smoke_test.sh`: call /health/ready, post one good and one defective fixture image, check both decisions. You run it locally now, and CI runs it against every built image in Phase 8.
6. Prove the image has no PyTorch and runs as non-root.

```bash
docker run --rm --entrypoint python <image> -c \
  "import sys, anomaly_inspector.serving.app; assert 'torch' not in sys.modules"
docker run --rm --entrypoint id <image>    # uid=10001(app)
```

7. Add a small demo page. Serve one static HTML file at `/` with an upload form that calls /v1/predict and shows the score, the decision and the heatmap. It adds no Python dependencies.
8. Publish and deploy once by hand, so you understand each step before automating it in Phase 8.

```bash
echo "$GHCR_TOKEN" | docker login ghcr.io -u <github-user> --password-stdin
docker tag anomaly-inspector-api ghcr.io/<github-user>/anomaly-inspector-api:0.1.0
docker push ghcr.io/<github-user>/anomaly-inspector-api:0.1.0
# make the package public in GitHub > Packages, then:
gcloud run deploy anomaly-inspector \
  --image ghcr.io/<github-user>/anomaly-inspector-api:0.1.0 \
  --port 8000 --cpu 2 --memory 1Gi --max-instances 1 \
  --region europe-west1 --allow-unauthenticated
```

### Where to host the demo

| Option | Why | Watch out for |
| --- | --- | --- |
| [Google Cloud Run](https://docs.cloud.google.com/run/docs/deploying) (recommended) | Runs public GHCR images directly, scales to zero, common in industry | Needs a Google Cloud billing account; keep `--max-instances 1` to cap cost |
| [Render](https://render.com/docs/free) free web service | No cost | Spins down after 15 minutes without traffic, so the first request is slow |
| [Hugging Face Spaces](https://huggingface.co/docs/hub/en/spaces-overview) | Popular for ML demos | Docker and Gradio Spaces now need a paid plan; free accounts are limited to static Spaces and up to 2 ZeroGPU Gradio Spaces |

### Pitfalls

- Different base images in the two stages. The virtual environment points at the builder's Python, so both stages must use the same tag.
- A health check that calls curl. Slim images do not include it; use Python, as above.
- Tagging only `latest`. Tag every image with the git SHA, and releases with a version.
- A public demo without limits. Cap instances and upload size; anyone can call it.

### Done when

- [ ] The API image builds, contains no torch, runs as uid 10001, and its size is in the README.
- [ ] `docker compose --profile mlflow --profile api up -d` shows every service healthy.
- [ ] The smoke test passes against the running container.
- [ ] The demo URL works from your phone, and it sits at the top of the README.
- [ ] MVP gate: update your CV and LinkedIn, and start applying while you build weeks 6 to 8.

## Phase 8: CI/CD with GitHub Actions

Week 6. **Goal:** no change reaches production unless the code passes its tests, the model passes its quality gates, and the image passes a smoke test and a vulnerability scan.

| Workflow | Runs on | What it does | Blocks the merge when |
| --- | --- | --- | --- |
| ci.yml | Every pull request and push to main | Lint, types, unit and API tests; builds the API image with a tiny model, smoke-tests it, scans it | Any check fails |
| model-ci.yml | Pull requests that touch ML code, params or data | Pulls data from R2, reruns the changed DVC stages on CPU, posts the metrics diff, runs the gates | A gate fails |
| cd.yml | Push to main and `v*` tags | Builds the image with the real model, pushes to GHCR, deploys to Cloud Run after your approval, smoke-tests the live URL | The live smoke test fails |

### Steps

1. Add a `docker` job to ci.yml. It builds the API with a tiny untrained model exported in the job, so it needs no data; the smoke test here checks status codes and the response schema, not decisions.

```yaml
  docker:
    runs-on: ubuntu-latest
    needs: checks
    steps:
      - uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7.0.1
      - uses: astral-sh/setup-uv@07a047e608db84b3d8c06cd9a7c7dcc9c355c18e # v10.3.2
      - run: uv sync --locked --extra train
      - run: uv run python -m anomaly_inspector.export.tiny_model --out models/ci/model.onnx
      - uses: docker/setup-buildx-action@f87e5991a6d7451dcb8d9637bfbc97413f497069 # v4.4.1
      - uses: docker/build-push-action@c3c9e263c25d99ce0380d002d59b67737d91b0dc # v7.4.0
        with:
          context: .
          file: docker/api.Dockerfile
          build-args: MODEL_FILE=models/ci/model.onnx
          tags: anomaly-inspector-api:ci
          load: true
          cache-from: type=gha
          cache-to: type=gha,mode=max
      - run: |
          docker run -d --name api -p 8000:8000 anomaly-inspector-api:ci
          ./scripts/smoke_test.sh http://localhost:8000
      - uses: anchore/scan-action@27805bf3b4e84b4a5c980df22ed233c00390a439 # v7.4.2
        with:
          image: anomaly-inspector-api:ci
          severity-cutoff: critical
          only-fixed: true
```

2. Write the gate as a small, tested script. Floors live in params.yaml under `gates`; set them about one point below your Phase 2 baseline. Also fail when ONNX and PyTorch AUROC differ, or when p95 latency exceeds the budget.

```python
# src/anomaly_inspector/pipelines/gate.py
import json
import sys
from pathlib import Path

from anomaly_inspector.config import load_params


def check(reports: Path, floors: dict[str, float], budget_ms: float) -> list[str]:
    failures = []
    for category in sorted(p.name for p in reports.iterdir() if p.is_dir()):
        m = json.loads((reports / category / "metrics.json").read_text())
        e = json.loads((reports / category / "export.json").read_text())
        failures += [f"{category}: {k}={m[k]:.3f} < {v}" for k, v in floors.items() if m[k] < v]
        if abs(e["onnx_image_auroc"] - e["torch_image_auroc"]) > 1e-3:
            failures.append(f"{category}: ONNX and PyTorch disagree")
        if e["latency_p95_ms"] > budget_ms:
            failures.append(f"{category}: p95 {e['latency_p95_ms']:.0f} ms > {budget_ms} ms")
    return failures


if __name__ == "__main__":
    gates = load_params().gates
    problems = check(Path("reports"), gates.floors, gates.latency_budget_ms)
    print("\n".join(problems) or "All gates passed")
    sys.exit(1 if problems else 0)
```

3. Add model-ci.yml. `dvc repro --pull` downloads cached outputs and recomputes only what the pull request changed; targeting `export` skips the slow STFPM stages. DVC reads the R2 keys from the standard AWS variables.

```yaml
name: model-ci
on:
  pull_request:
    paths: ["src/anomaly_inspector/**", "params.yaml", "dvc.yaml", "dvc.lock", "data/**/*.dvc"]
permissions:
  contents: read
  pull-requests: write
jobs:
  pipeline:
    runs-on: ubuntu-latest
    timeout-minutes: 60
    env:
      AWS_ACCESS_KEY_ID: ${{ secrets.R2_ACCESS_KEY_ID }}
      AWS_SECRET_ACCESS_KEY: ${{ secrets.R2_SECRET_ACCESS_KEY }}
    steps:
      - uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7.0.1
        with:
          fetch-depth: 0  # dvc metrics diff needs main's history
      - uses: astral-sh/setup-uv@07a047e608db84b3d8c06cd9a7c7dcc9c355c18e # v10.3.2
      - run: uv sync --locked --extra train
      - run: uv run dvc repro --pull export
      - run: uv run dvc metrics diff origin/main --md > metrics_diff.md
      - uses: marocchino/sticky-pull-request-comment@5770ad5eb8f42dd2c4f34da00c94c5381e49af88 # v3.0.5
        with:
          path: metrics_diff.md
      - run: uv run python -m anomaly_inspector.pipelines.gate
```

4. Add cd.yml. The image gets the real model from DVC, so production always matches a git commit. `uvx` runs DVC in a throwaway environment, without installing PyTorch. Google Cloud access uses OIDC through Workload Identity Federation, so no cloud key is stored in GitHub; set it up from the [auth action README](https://github.com/google-github-actions/auth) and restrict it to your repository.

```yaml
name: cd
on:
  push:
    branches: [main]
    tags: ["v*"]
permissions:
  contents: read
  packages: write
  id-token: write
jobs:
  release:
    runs-on: ubuntu-latest
    environment: production  # required reviewer = you: one click to approve
    env:
      IMAGE: ghcr.io/${{ github.repository_owner }}/anomaly-inspector-api
      AWS_ACCESS_KEY_ID: ${{ secrets.R2_ACCESS_KEY_ID }}
      AWS_SECRET_ACCESS_KEY: ${{ secrets.R2_SECRET_ACCESS_KEY }}
    steps:
      - uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7.0.1
      - uses: astral-sh/setup-uv@07a047e608db84b3d8c06cd9a7c7dcc9c355c18e # v10.3.2
      - run: uvx --from "dvc[s3]==3.67.1" dvc pull models/patchcore/bottle/model.onnx
      - uses: docker/setup-buildx-action@f87e5991a6d7451dcb8d9637bfbc97413f497069 # v4.4.1
      - uses: docker/login-action@dbcb813823bdd20940b903addbd779551569679f # v4.6.0
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      - id: meta
        uses: docker/metadata-action@dc802804100637a589fabce1cb79ff13a1411302 # v6.2.0
        with:
          images: ${{ env.IMAGE }}
          tags: |
            type=sha,format=long
            type=semver,pattern={{version}}
      - uses: docker/build-push-action@c3c9e263c25d99ce0380d002d59b67737d91b0dc # v7.4.0
        with:
          context: .
          file: docker/api.Dockerfile
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
      - uses: google-github-actions/auth@7c6bc770dae815cd3e89ee6cdf493a5fab2cc093 # v3.0.0
        with:
          workload_identity_provider: ${{ vars.GCP_WIF_PROVIDER }}
          service_account: ${{ vars.GCP_DEPLOY_SA }}
      - id: deploy
        uses: google-github-actions/deploy-cloudrun@2028e2d7d30a78c6910e0632e48dd561b064884d # v3.0.1
        with:
          service: anomaly-inspector
          region: europe-west1
          image: ${{ env.IMAGE }}:sha-${{ github.sha }}
          flags: --port=8000 --max-instances=1
      - run: ./scripts/smoke_test.sh "${{ steps.deploy.outputs.url }}" --check-decisions
```

5. Protect main: require the `checks`, `docker` and `pipeline` checks before merging. Roll back by moving traffic to the previous Cloud Run revision with `gcloud run services update-traffic`.
6. Harden the workflows. Each point below is a common interview question.

- **Pin every action to a full commit SHA.** Tags can be moved. On 19 March 2026, attackers force-pushed 76 of 77 version tags of `aquasecurity/trivy-action` to malicious commits; the payload dumped runner memory for secrets and ran before the real scan, so workflows still looked green ([Aqua Security](https://www.aquasec.com/blog/trivy-supply-chain-attack-what-you-need-to-know)). Use [pinact](https://github.com/suzuki-shunsuke/pinact) to pin, and Dependabot to update the SHAs.
- **Least privilege.** Declare `permissions:` in every workflow; the default token should only read.
- **No `pull_request_target` with a checkout of pull request code.** It hands untrusted code your secrets.
- **No long-lived cloud keys.** OIDC for Google Cloud; a bucket-scoped token for R2.
- **Dependabot** for the `github-actions`, `uv` and `docker` ecosystems, weekly.

### Pitfalls

- `dvc metrics diff main` failing in CI. Checkout is shallow and has no local main branch; use `fetch-depth: 0` and `origin/main`.
- Retraining everything on every pull request. The `paths` filter and `dvc repro --pull` keep model-ci to the stages that changed.
- Pull requests from forks get no secrets, so model-ci cannot pull data for them. That is fine for a solo repo; say so in CONTRIBUTING.md.

### Done when

- [ ] Every pull request shows green `checks` and `docker`, plus `pipeline` when ML code changed.
- [ ] A deliberately bad pull request (coreset ratio 0.0001) gets a metrics comment and a red gate. Close it unmerged.
- [ ] A merge to main deploys a new revision after your approval, and the live smoke test passes.
- [ ] Every action is pinned by SHA, and Dependabot is on.
- [ ] The README shows the three workflow badges.

## Phase 9: Orchestration with Airflow 3

Week 7. **Goal:** two DAGs that run without you. `retrain_patchcore` turns confirmed new images into a challenger and proposes it through a pull request; `batch_inspection` scores a simulated production line every 10 minutes.

The key design rule: Airflow orchestrates, it does not compute. Every ML step runs in your training image through DockerOperator, so the Airflow image never needs PyTorch or your package. And Airflow never deploys directly: a winning challenger becomes a pull request, CI runs the gates from Phase 8, and merging deploys.

### Steps

1. Give Airflow its own clone of the repo, separate from the folder you edit; its git commands would otherwise switch your branch.

```bash
mkdir -p ~/airflow-workspace/{incoming,line/incoming,review,archive}
git clone git@github.com:<github-user>/industrial-anomaly-detection.git ~/airflow-workspace/industrial-anomaly-detection
cd ~/airflow-workspace/industrial-anomaly-detection && uv run dvc pull
```

2. Simulate the factory. Before Phase 1, move 20% of each category's train/good images to a `data/simulation/` pool that training never sees. `scripts/simulate_line.py` copies a mix of pool and defect images into `line/incoming/` every few minutes, and confirmed good ones into `incoming/<category>/good/`.
3. Start Airflow from the official Compose file for 3.3.2 ([Running Airflow in Docker](https://airflow.apache.org/docs/apache-airflow/stable/howto/docker-compose/index.html)). The docs ask for at least 4 GB of memory for Docker.

```bash
mkdir -p airflow && cd airflow
curl -LfO 'https://airflow.apache.org/docs/apache-airflow/3.3.2/docker-compose.yaml'
mkdir -p ./dags ./logs ./plugins ./config
echo -e "AIRFLOW_UID=$(id -u)" > .env
docker compose up airflow-init
docker compose up -d       # UI on http://localhost:8080, login airflow / airflow
```

4. Edit that Compose file for your laptop:

- Switch to `LocalExecutor` and delete the redis, airflow-worker and flower services; that saves memory, and tasks then run in the scheduler container.
- Mount `/var/run/docker.sock` into the scheduler, and add your host's docker group id with `group_add` (`getent group docker` shows it).
- Mount `~/airflow-workspace` at the same absolute path inside Airflow. DockerOperator mount paths are host paths, so identical paths remove a whole class of bugs.
- Join one shared network. In your main docker-compose.yml set `networks: {default: {name: inspection}}`; in Airflow's file declare the same network with `external: true`. Containers then reach `mlflow:5000` and `api:8000` by name.
- Put `INSPECTION_WORKSPACE`, the R2 keys and a fine-grained GitHub token (contents and pull requests on this repo only) in Airflow's .env.

5. Write the retraining DAG. ML steps run in the training image; only light decisions run in Airflow.

```python
# airflow/dags/retrain_patchcore.py
import json
import os
from datetime import datetime
from pathlib import Path

from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sdk import dag, task
from docker.types import Mount

WORKSPACE = os.environ.get("INSPECTION_WORKSPACE", "/tmp/workspace")
REPO = f"{WORKSPACE}/industrial-anomaly-detection"
CATEGORIES = ["bottle", "carpet", "screw"]
MIN_NEW_IMAGES = 20


def in_train_image(task_id: str, command: str) -> DockerOperator:
    return DockerOperator(
        task_id=task_id,
        image="anomaly-inspector-train:latest",
        command=["bash", "-c", command],
        working_dir="/repo",
        mounts=[Mount(source=REPO, target="/repo", type="bind")],
        network_mode="inspection",
        environment={"MLFLOW_TRACKING_URI": "http://mlflow:5000", "PYTHONPATH": "/repo/src"},
        private_environment={  # hidden in the UI
            "AWS_ACCESS_KEY_ID": os.environ.get("R2_ACCESS_KEY_ID", ""),
            "AWS_SECRET_ACCESS_KEY": os.environ.get("R2_SECRET_ACCESS_KEY", ""),
            "GH_TOKEN": os.environ.get("GH_TOKEN", ""),
        },
        mount_tmp_dir=False,  # Airflow itself runs in Docker
        auto_remove="success",
        cpus=8.0,  # the default is 1.0
        mem_limit="8g",
    )


@dag(schedule="@weekly", start_date=datetime(2026, 1, 1), catchup=False, tags=["retraining"])
def retrain_patchcore():
    @task.short_circuit
    def new_data_ready() -> bool:
        incoming = Path(WORKSPACE, "incoming")
        return any(len(list((incoming / c / "good").glob("*.png"))) >= MIN_NEW_IMAGES for c in CATEGORIES)

    ingest = in_train_image("ingest", "python -m anomaly_inspector.pipelines.ingest && dvc add data/raw/mvtec_ad")
    retrain = in_train_image("retrain", "dvc repro --pull export")
    register = in_train_image("register", "python -m anomaly_inspector.pipelines.register --all")
    # register prints one JSON line last, e.g. {"bottle": true, "carpet": false}; it becomes the XCom

    @task.branch
    def decide(promoted_json: str) -> str:
        return "open_pull_request" if any(json.loads(promoted_json).values()) else "no_change"

    open_pr = in_train_image("open_pull_request", "bash scripts/open_retrain_pr.sh {{ ds }}")

    @task
    def no_change() -> None:
        print("No challenger beat its champion. Nothing to deploy.")

    new_data_ready() >> ingest >> retrain >> register
    decide(register.output) >> [open_pr, no_change()]


retrain_patchcore()
```

`open_retrain_pr.sh` creates a `retrain/<date>` branch, commits the new `.dvc` file and `dvc.lock`, runs `dvc push`, pushes the branch and opens a pull request through the GitHub REST API.

6. Write the batch inspection DAG. It calls the API over HTTP, as a real line PC would, and uses dynamic task mapping to score chunks in parallel.

```python
# airflow/dags/batch_inspection.py
@dag(schedule="*/10 * * * *", start_date=datetime(2026, 1, 1), catchup=False,
     max_active_runs=1, tags=["inspection"])
def batch_inspection():
    @task
    def list_new_images() -> list[list[str]]:
        files = sorted(str(p) for p in Path(WORKSPACE, "line", "incoming").glob("*.png"))
        return [files[i : i + 25] for i in range(0, len(files), 25)]

    @task(retries=3, retry_exponential_backoff=True)
    def score_chunk(paths: list[str]) -> list[dict]:
        import httpx

        results = []
        with httpx.Client(base_url="http://api:8000", timeout=30) as client:
            for p in paths:
                with open(p, "rb") as f:
                    r = client.post("/v1/predict", files={"file": (Path(p).name, f, "image/png")})
                r.raise_for_status()
                results.append({"path": p, **r.json()})
        return results

    @task
    def store_and_route(chunks: list[list[dict]]) -> None:
        """Append results to a dated Parquet log first; only then move each file to review/ or archive/."""

    store_and_route(score_chunk.expand(paths=list_new_images()))


batch_inspection()
```

7. Test the DAGs with pytest. Airflow pins many libraries, so run these tests in the official image rather than in your project environment.

```python
# tests/airflow/test_dags.py
import pytest

pytest.importorskip("airflow")
from airflow.dag_processing.dagbag import DagBag


@pytest.fixture(scope="session")
def dagbag() -> DagBag:
    return DagBag(dag_folder="airflow/dags", include_examples=False)


def test_no_import_errors(dagbag):
    assert dagbag.import_errors == {}


def test_retrain_has_gate_and_branch(dagbag):
    dag = dagbag.dags["retrain_patchcore"]
    assert {"new_data_ready", "retrain", "register", "decide"} <= set(dag.task_ids)
```

```bash
docker run --rm -v "$PWD:/repo:ro" -w /repo apache/airflow:3.3.2 \
  bash -c "pip install --user pytest && python -m pytest -p no:cacheprovider tests/airflow"
```

Add the same command as an `airflow-dags` job in ci.yml.

### Pitfalls

- Two DVC commands at once in the same repo fail with a lock error. Keep the DVC tasks sequential; map only HTTP or MLflow calls.
- Top-level code that touches the database, the network or heavy imports. Airflow parses DAG files constantly, so keep that work inside tasks ([best practices](https://airflow.apache.org/docs/apache-airflow/stable/best-practices.html)).
- Training on the model's own guesses. Only human-confirmed or known-good images go to `incoming/`; otherwise the model learns its own mistakes.
- Non-idempotent tasks. Save results before moving files, so a retry never loses or double-counts an image.
- Passing images through XCom. Pass paths, never pixels.

### Done when

- [ ] Both DAGs show in the UI with no import errors.
- [ ] 20 confirmed images in `incoming/bottle/good/` lead to a retrain and, if the challenger wins, a pull request that CI checks.
- [ ] A losing challenger ends in `no_change`, and nothing is deployed.
- [ ] batch\_inspection runs every 10 minutes; flagged images land in review/ and every result is in the Parquet log.
- [ ] The DAG tests run in CI.

## Phase 10: Monitoring and the feedback loop

Week 8, days 1 to 3. **Goal:** one Grafana dashboard shows service health and model health, a daily check flags when the line changes, and review decisions flow back into the data.

A deployed model fails quietly: the API stays green while a new light or a dirty lens shifts every score. Watch four kinds of signal.

| Signal | What to measure | What it catches |
| --- | --- | --- |
| Service | Request rate, error rate, p50 and p95 latency | API outages and slowdowns |
| Model output | Score histogram, share of parts flagged | Process changes or a broken model |
| Input | Brightness, contrast, sharpness (variance of the Laplacian) | Camera and lighting drift, often first |
| Delayed ground truth | Review outcomes: confirmed defects vs false alarms | Real accuracy in production |

### Steps

1. Add custom metrics to the API. Record the score divided by the threshold, so one set of buckets works for every category and model.

```python
from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator

SCORE = Histogram(
    "anomaly_score_ratio", "Image score divided by the threshold", ["category"],
    buckets=[0.25, 0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0, 3.0],
)
DECISIONS = Counter("inspection_decisions_total", "Decisions", ["category", "result"])

Instrumentator().instrument(app).expose(app)  # /metrics: request counts and latency

# inside predict():
SCORE.labels(inspector.category).observe(pred.score / inspector.threshold)
DECISIONS.labels(inspector.category, "defect" if pred.is_anomalous else "ok").inc()
```

2. Add Prometheus and Grafana to Compose under a `monitoring` profile: `prom/prometheus:v3.15.1` scraping `api:8000`, and `grafana/grafana:13.2.4` with its data source and dashboard provisioned from `monitoring/grafana/` in the repo. Dashboards as code means a fresh clone gets the same dashboard.

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s
rule_files: [/etc/prometheus/alerts.yml]
scrape_configs:
  - job_name: api
    static_configs:
      - targets: ["api:8000"]
```

3. Build the dashboard and two alert rules from these queries.

```text
# p95 latency of /v1/predict
histogram_quantile(0.95, sum by (le) (rate(http_request_duration_seconds_bucket{handler="/v1/predict"}[5m])))

# share of parts flagged over 30 minutes; alert when it doubles the expected rate
sum(rate(inspection_decisions_total{result="defect"}[30m])) / sum(rate(inspection_decisions_total[30m]))
```

4. Save a reference at training time. The evaluate stage writes `reference.parquet`: scores and image statistics on val\_normal and holdout. It is a DVC output, so it is versioned with the model it describes.
5. Add a daily `drift_report` DAG that runs in the training image. It compares the last 24 hours of the inspection log with the reference: a Kolmogorov-Smirnov test (`scipy.stats.ks_2samp`) and the population stability index on scores and image statistics, plus the flagged rate. It writes `reports/drift/<date>.md` and fails the task, which alerts, above your limits.

```python
import numpy as np


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population stability index. A common rule of thumb: above 0.25 is a large shift."""
    edges = np.quantile(reference, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    ref = np.clip(np.histogram(reference, edges)[0] / len(reference), 1e-6, None)
    cur = np.clip(np.histogram(current, edges)[0] / len(current), 1e-6, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))
```

6. Test the monitoring on purpose. Make the simulator darken images by 30% for an hour. The flagged rate should rise on the dashboard, and the next drift report should alert. Write down how long detection took; it is one of the best stories in this project.
7. Close the loop with human review. Flagged images wait in review/. A small page served by the API, or a tool such as Label Studio, records each decision (true defect or false alarm) to a reviews table.

- False alarms that a person confirms as good can go to `incoming/<category>/good/`, which feeds the retraining DAG.
- Confirmed defects go to a growing calib set for threshold recalibration. Keep the holdout frozen.
- Production precision (confirmed defects divided by flagged parts) goes into the weekly report.

### Pitfalls

- Alerts on single images. Alert on rates over a window, or the alarm rings all day.
- Several uvicorn workers with prometheus\_client. Each process keeps its own counters, so a scrape sees only one. Run one worker per container, or configure the client's multiprocess mode.
- High-cardinality labels such as an image id. Use labels with few values only: category and result.
- A reference computed from the wrong data. Build it at training time from val\_normal and holdout, and version it with the model.

### Done when

- [ ] The provisioned dashboard shows request rate, p95 latency, error rate, the score histogram and the flagged rate.
- [ ] Alert rules exist for the flagged rate and p95 latency.
- [ ] The drift DAG writes a daily report, and the simulated lighting change triggers it.
- [ ] Review decisions are stored, and the weekly report shows production precision.
- [ ] A dashboard screenshot from the drift test is in the README.

## Portfolio polish

Week 8, days 4 and 5. **Goal:** a recruiter understands the project in 30 seconds, and an engineer can run it in 10 minutes. Write the README, model card and decision records in your own words; interviewers will ask about every decision, and writing them yourself is how you get ready.

### README, in this order

```markdown
# Industrial Anomaly Inspector
Detects and localizes defects on manufactured parts from defect-free training images only.
[CI badge] [model-ci badge] [CD badge] [license badge]

![demo](docs/demo.gif)  Live demo: <url>

## Results (MVTec AD holdout split, CPU: <model>, <n> threads)
| Category | Image AUROC | AUPRO | Miss rate | False rejects | p95 latency |

## Architecture
<diagram: Mermaid renders on GitHub>

## Quickstart
make setup && uv run dvc pull && uv run dvc repro
docker compose --profile mlflow --profile api up -d

## How it works
Model, evaluation protocol, threshold choice, MLOps lifecycle

## Design decisions
Links to docs/adr/

## Limitations and next steps

## Data and license
MVTec AD is CC BY-NC-SA 4.0, non-commercial. Code is MIT.
```

### Model card and data card

Write `docs/model_card.md` with the sections from [Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993): intended use (assisting visual inspection on these three products, with a person reviewing flagged parts), training data and license, evaluation protocol and holdout metrics per defect type, the operating point and its cost assumption, known failure modes from your gallery, and limits such as lighting changes and unseen defect types. The data card from Phase 1 sits next to it.

### Decision records

Write one short file per decision in `docs/adr/`, each with three parts: context, decision, consequences. These five cover the questions interviewers ask most:

1. PatchCore in production, STFPM as the fast alternative, with your accuracy and latency numbers.
2. ONNX Runtime for serving, with no PyTorch in the image.
3. DVC for data and pipeline, MLflow for runs and the registry.
4. A threshold calibrated on held-out data, with a stated 10:1 cost ratio.
5. Airflow proposes, CI decides: retraining reaches production only through a pull request.

### CV bullets

Templates only. Replace every bracket with a number you measured; never round up or invent one.

- Built an end-to-end visual inspection system in PyTorch, implementing PatchCore from the paper; reached \[X\]% image AUROC and \[Y\]% AUPRO on a held-out MVTec AD split, with the threshold calibrated on separate data.
- Served the model with FastAPI and ONNX Runtime in a \[N\] MB Docker image without PyTorch; INT8 quantization cut CPU p95 latency from \[A\] ms to \[B\] ms for a \[C\]-point AUROC change.
- Automated the lifecycle with DVC, an MLflow registry with champion and challenger aliases, Airflow retraining, and GitHub Actions gates that block pull requests that lower accuracy or exceed the latency budget.
- Added Prometheus and Grafana monitoring and daily drift checks that caught a simulated lighting failure within \[T\] minutes.

### Get it seen

- Post short weekly progress updates on LinkedIn during the 8 weeks, each with one picture: a heatmap, a chart, a green pipeline.
- At the end, publish one longer write-up with the demo GIF, the failure gallery and the drift test, and link it from the README.
- Record a 2-minute walkthrough video: demo, architecture, one failure case, one design decision.
- Pin the repo on your GitHub profile.

### Done when

- [ ] A friend who has never seen the repo runs the quickstart in under 10 minutes.
- [ ] The README shows the demo, live link, results table, diagram and badges at the top.
- [ ] The model card, data card and five decision records are written in your own words.
- [ ] gitleaks finds no secrets in the history, and the code has a LICENSE file.
- [ ] Your CV and LinkedIn show the project with your real numbers.

## Interview prep

Every question below has an answer in your repo. Practice each one aloud in under 90 seconds, and always include one number you measured.

| Question | Answer with | Show |
| --- | --- | --- |
| Why not a supervised classifier? | Defects are rare and new types keep appearing; training on good parts catches unknown defects. Supervised models come later, from review labels. | Problem framing, review loop |
| How does PatchCore work, and why the coreset? | Mid-level patch features, a memory bank of normal patches, distance to the nearest one. The coreset keeps coverage with about 1% of patches, so search is faster and the model smaller. | patchcore.py, coreset experiment |
| PatchCore or STFPM in production? | Your accuracy, latency and size numbers. The memory bank grows with data; STFPM has a fixed size but needs training. | Decision record 1 |
| Why AUPRO and not only pixel AUROC? | Pixel AUROC is dominated by large defects and millions of normal pixels. AUPRO counts every defect region equally. | metrics.py and its tests |
| How did you choose the threshold? | On the calib split with a 10:1 cost ratio, plus a cold-start version from normal scores only. Reported on holdout, never tuned on it. | Decision record 4, threshold.json |
| Where does your model fail? | Two or three cases from your gallery, and why each one fails. | Failure gallery |
| Isn't DVC plus MLflow redundant? | The ownership table from Phase 4. | dvc.yaml, MLflow UI |
| How do you know serving matches training? | One preprocessing function with a golden test, ONNX parity on outputs and AUROC, and the no-torch import guard. | Tests |
| What stops a bad model from shipping? | model-ci reruns the changed stages, posts the diff, and the gate fails. CD also needs your approval and a live smoke test. | The closed bad pull request |
| How does retraining work, and who approves it? | Airflow proposes through a pull request, CI decides, you merge, CD deploys. Rollback is a traffic shift to the previous revision. | retrain\_patchcore DAG |
| How would you notice a problem without labels? | Shifts in scores and flagged rate, input drift (PSI, KS test), then delayed review labels. Quote your drift-test detection time. | Dashboard screenshot, drift report |
| Why ONNX Runtime instead of serving PyTorch? | Smaller image, faster start, INT8 on CPU, and a path to OpenVINO or TensorRT. Quote your benchmark. | Benchmark table |
| How did you secure the pipeline? | SHA-pinned actions after the March 2026 Trivy incident, least-privilege tokens, OIDC, a non-root image, upload validation. | Workflows, Dockerfile |
| How would you scale to 50 product lines? | Per-product memory banks and thresholds, or one multi-class model such as Dinomaly; the same pipeline through `foreach`; monitoring labelled by category. | Your stretch notes |
| What would you do next? | MVTec AD 2 or VisA, a multi-class model, edge deployment with OpenVINO, active learning from review labels. | README limitations |

Prepare two stories in situation, task, action, result form: a bug a test caught (for example a preprocessing mismatch), and the lighting-drift test. For live coding, be ready to write a Dataset and training loop, compute AUROC by hand, and debug a tensor shape error; you will have done all three in Phase 2.

## Resources

### Papers

- [PatchCore](https://arxiv.org/abs/2106.08265): the main model, CVPR 2022
- [STFPM](https://arxiv.org/abs/2103.04257): the student-teacher model you train
- [EfficientAD](https://arxiv.org/abs/2303.14535): a fast student-teacher model, a good next benchmark
- [Dinomaly](https://arxiv.org/abs/2405.14325): one model for many categories
- [SPot-the-Difference](https://arxiv.org/abs/2207.14315): the paper that introduced VisA
- [MVTec AD 2](https://arxiv.org/abs/2503.21622): the harder benchmark
- [Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993)

### Code to learn from

- [anomalib](https://github.com/open-edge-platform/anomalib): reference implementations and benchmarks
- [Official PatchCore code](https://github.com/amazon-science/patchcore-inspection)
- [awesome-industrial-anomaly-detection](https://github.com/M-3LAB/awesome-industrial-anomaly-detection): papers and datasets, kept up to date by the community

### Documentation

- [uv with PyTorch](https://docs.astral.sh/uv/guides/integration/pytorch/) and [uv in Docker](https://docs.astral.sh/uv/guides/integration/docker/)
- [DVC](https://dvc.org/doc)
- [MLflow tracking server](https://mlflow.org/docs/latest/self-hosting/architecture/tracking-server/) and [network security](https://mlflow.org/docs/latest/self-hosting/security/network/)
- [PyTorch ONNX export](https://docs.pytorch.org/docs/2.14/onnx_export.html) and [ONNX Runtime quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)
- [FastAPI testing](https://fastapi.tiangolo.com/tutorial/testing/)
- [Airflow in Docker](https://airflow.apache.org/docs/apache-airflow/stable/howto/docker-compose/index.html), [Airflow best practices](https://airflow.apache.org/docs/apache-airflow/stable/best-practices.html) and [DockerOperator](https://airflow.apache.org/docs/apache-airflow-providers-docker/stable/_api/airflow/providers/docker/operators/docker/index.html)
- [Secure use of GitHub Actions](https://docs.github.com/en/actions/reference/security/secure-use), [Google auth action](https://github.com/google-github-actions/auth) and [pinact](https://github.com/suzuki-shunsuke/pinact)

### Courses and books

- [Made With ML](https://madewithml.com): a free MLOps course
- [MLOps Zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp): free and project-based
- Designing Machine Learning Systems, by Chip Huyen (O'Reilly, 2022)

### Versions checked on PyPI, 22 October 2026

Pin these in uv.lock and update them on purpose, through Dependabot pull requests.

| Package | Version | Released |
| --- | --- | --- |
| anomalib | 2.7.0 | 2026-10-06 |
| apache-airflow | 3.3.2 | 2026-09-17 |
| dvc | 3.67.1 | 2026-03-31 |
| fastapi | 0.159.1 | 2026-10-21 |
| mlflow | 3.17.2 | 2026-10-21 |
| onnx | 1.23.2 | 2026-10-06 |
| onnxruntime | 1.31.0 | 2026-10-09 |
| pytest | 9.1.1 | 2026-06-19 |
| ruff | 0.17.1 | 2026-10-15 |
| torch | 2.14.1 | 2026-09-30 |
| torchvision | 0.29.1 | 2026-09-30 |
| uv | 0.13.5 | 2026-10-22 |

### Other sources cited in this guide

- Datasets: [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad), [MVTec AD downloads](https://www.mvtec.com/company/research/datasets/mvtec-ad/downloads), [VisA](https://github.com/amazon-science/spot-diff), [MVTec AD 2](https://www.mvtec.com/company/research/datasets/mvtec-ad-2)
- Hosting and storage: [Cloudflare R2 pricing](https://developers.cloudflare.com/r2/pricing/), [Cloud Run deploying](https://docs.cloud.google.com/run/docs/deploying), [Render free tier](https://render.com/docs/free), [Hugging Face Spaces](https://huggingface.co/docs/hub/en/spaces-overview)
- Ecosystem news: [DVC joins lakeFS](https://dvc.org/blog/dvc-joins-lakefs-your-questions-answered/), [Trivy supply chain attack](https://www.aquasec.com/blog/trivy-supply-chain-attack-what-you-need-to-know)
