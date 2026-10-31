# Anomaly Inspector

[![ci](https://github.com/you-ssef-dan/anomaly-inspector/actions/workflows/ci.yml/badge.svg)](https://github.com/you-ssef-dan/anomaly-inspector/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.12-blue)
![license](https://img.shields.io/badge/code-Apache--2.0-green)

**Visual inspection for factory parts: it learns what a good part looks like, flags the ones that are
not, and draws a heatmap showing where the defect is.** Built end to end on a CPU laptop, from raw
images to a tested, versioned pipeline.

**Live demo: [anomaly-inspector-5fe2.onrender.com](https://anomaly-inspector-5fe2.onrender.com)**.
Upload a photo of a bottle and see the decision and the heatmap ([how it is deployed](#live-on-render)).
It runs on a free plan, so the first visit can take about a minute to wake it up.

> **Status: work in progress.** Data, models, metrics, experiment tracking, the DVC pipeline, the
> ONNX export, the API, its Docker image (public on GHCR) and a live demo are done; CI/CD gates,
> retraining and monitoring come next. See the [roadmap](#roadmap).

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
(bottle, carpet, screw). This repo's PatchCore (WideResNet-50, 256 px, 1% memory bank), scored on
`holdout`, the split that no decision has touched:

| Holdout | Images (good + defective) | Image AUROC | Pixel AUROC | AUPRO | ms per image |
| --- | --- | --- | --- | --- | --- |
| Bottle | 10 + 32 | 1.0000 | 0.9862 | 0.9410 | 99 |
| Carpet | 14 + 45 | 0.9873 | 0.9896 | 0.9420 | 104 |
| Screw | 21 + 59 | 0.8918 | 0.9778 | 0.8852 | 107 |

- **AUPRO** counts every defect region equally, however small, so it is the metric that cares about
  tiny defects. Screw is the hard category: tiny defects on parts lying at any angle.
- Each row comes from the DVC pipeline (`dvc repro`), which writes it to
  `reports/patchcore/<category>/metrics.json`; the model is registered in MLflow as the current
  `champion` of its category (see [tracking](#tracking-and-model-registry)).
- The holdout sets are small: with 21 good screws, a single image can move image AUROC by up to
  1/21, about 5 points.
- Latency is the model alone, on a laptop CPU (Intel Core Ultra 5 125H), batch 8. It depends on
  what else the machine is doing: the same screw model measured 169 ms during the experiments.

**Checked against anomalib.** On bottle's full test set, with identical inputs, splits, coreset size,
seed and metric code (anomalib's backbone gets the same ImageNet weights), the two implementations
differ by less than 0.001:

| Bottle, full test set (83 images) | Image AUROC | Pixel AUROC | AUPRO |
| --- | --- | --- | --- |
| PatchCore (this repo) | 1.0000 | 0.9866 | 0.9413 |
| anomalib PatchCore | 1.0000 | 0.9868 | 0.9417 |

- Our AUPRO code agrees with anomalib's to 5 decimals on the same maps (a slow test checks it).
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

## Tracking and model registry

When a server is configured, every evaluation in the pipeline is logged to
[MLflow](https://mlflow.org) (server and PostgreSQL in Docker Compose), so any number above can be
traced back to what produced it:

| Logged | What |
| --- | --- |
| Tags | git SHA, whether the code had uncommitted changes, DVC data hash, md5 of the model file, category, CPU model |
| Params | every value of `params.yaml` |
| Metrics | holdout image AUROC, pixel AUROC, AUPRO, ms per image; CPU and RAM over time |
| Artifacts | `threshold.json`, ROC and PRO curves, the holdout failure gallery |
| Model | the ONNX file that serves, registered as a new version of `patchcore-<category>` by `scripts/register_model.py`, outside the pipeline, so re-running a stage never adds a version |

**Promotion is code, not a click.** Each new version becomes the `challenger`.
[`should_promote`](src/anomaly_inspector/registry.py) makes it the `champion` only if it was trained
from committed code, stays under the latency budget (200 ms per image), and is not worse than the
current champion: image AUROC not lower, AUPRO at most 0.005 lower. Anything downstream loads
`models:/patchcore-screw@champion`, never a version number. Reloading each champion in a fresh
session reproduces its logged holdout AUROC exactly.

The `threshold.json` of each run holds the decision threshold, chosen on `calib` with a missed
defect assumed to cost 10 times a false reject, and, for comparison, the 99th percentile of good
parts' scores (the option for a new line that has no defect images yet).

The Phase 2 experiments are in MLflow too. All 19 screw runs, one line each, coloured by AUPRO: the
dark red line is ResNet-18 at 320 px, both the fastest run and the most accurate.

![The 19 screw experiments in MLflow's parallel coordinates view](reports/figures/mlflow_screw_runs.png)

And how many good images a new line needs (3 random subsets per count): with 1 to 10 images AUPRO
stays below 0.73; with 50 it is 0.845 to 0.882 (all 230 images: 0.865).

![AUPRO against the number of good training images, in MLflow](reports/figures/mlflow_images.png)

## From PyTorch to ONNX

The pipeline exports each model to [ONNX](https://onnx.ai): one file that runs in ONNX Runtime
without PyTorch, and carries its own threshold, image size, category and git SHA. The export
stage scores the ONNX model on `holdout` and fails if it differs from PyTorch by 0.001 or more;
it doesn't (same image AUROC, AUPRO within 0.0000003).

A second stage tries **INT8**: the convolutions are stored and computed in 8 bits, calibrated on
100 good `train` images; the distances to the memory bank stay in 32-bit floats. Then all three
versions are timed the same way. Screw:

| Screw | ms per image, batch 1 (p50 / p95) | Images per s, batch 8 | File | Image AUROC | AUPRO |
| --- | --- | --- | --- | --- | --- |
| PyTorch FP32 | 108 / 127 | 8.5 | 114 MB | 0.8918 | 0.8852 |
| ONNX Runtime FP32 | 90 / 93 | 11.2 | 114 MB | 0.8918 | 0.8852 |
| ONNX Runtime INT8 | 77 / 78 | 13.3 | 40 MB | 0.8870 | **0.8538** |

- **ONNX FP32 is free speed**: identical scores, about 1.2× faster per image, a much steadier p95,
  and a third more images per second in batches. The same holds for bottle and carpet.
- **INT8 is 3× smaller and about 15% faster again, but it blurs the heatmap.** It still tells good
  from defective parts (image AUROC within 0.005), yet AUPRO drops on every category: screw 0.885 →
  0.854, bottle 0.941 → 0.764, carpet 0.942 → **0.550**. The heatmap is half of what this system
  delivers, so **the FP32 ONNX model is the one that serves**; INT8 stays a measured option.
- Intel Core Ultra 5 125H, 4 threads for every version, on mains power with nothing else running;
  the model only, after 5 warm-up calls, 30 timed calls each. All numbers:
  `reports/patchcore/<category>/benchmark.json`.

## The API

A [FastAPI](https://fastapi.tiangolo.com) service answers "is this part defective, and where?". It
serves the ONNX model with ONNX Runtime, so it needs no PyTorch (a test checks that it never
loads it), and reads the threshold and image size from the model file.

| Method | Path | Answer |
| --- | --- | --- |
| GET | `/health/live` | The process is up |
| GET | `/health/ready` | The model is loaded (503 until then) |
| GET | `/v1/model` | Category, threshold, image size, model version |
| POST | `/v1/predict` | One photo: anomaly score, decision and, with `?heatmap=true`, the heatmap |
| POST | `/v1/predict/batch` | Up to 8 photos in one model call |
| GET | `/` | A demo page: pick a photo, see the decision, the score against the threshold and the heatmap |

```bash
API_MODEL_PATH=models/patchcore/screw/model.onnx uv run uvicorn anomaly_inspector.serving.app:app
curl -F "file=@part.png;type=image/png" localhost:8000/v1/predict   # or open localhost:8000/docs
```

A photo that is not PNG or JPEG gets a 415, one above 5 MB a 413, one that cannot be decoded a
422; each error names the file. The model loads once, at startup, and a model without its
metadata stops the startup. The tests run the endpoints against a fake model, and end to end
against a small real one; slow tests check the real bottle model (a good part passes, a broken
one is rejected, a black square pasted on a good part raises its score).

### Load test with Locust

[Locust](https://locust.io) simulates many clients calling the API at the same time and measures
response times, throughput and errors. Here each simulated user is a camera that sends a screw
photo from `holdout` about once a second ([`tests/load/locustfile.py`](tests/load/locustfile.py)).
The API runs on 2 CPU cores, the size of the container in the next phase:

```bash
API_MODEL_PATH=models/patchcore/screw/model.onnx API_THREADS=1 \
  taskset -c 0,1 uv run uvicorn anomaly_inspector.serving.app:app --port 8000 --workers 2
uv run locust -f tests/load/locustfile.py --host http://localhost:8000 \
  --headless -u 10 -r 2 -t 2m --csv reports/load/w2-t1
```

| On 2 cores | Cameras | Photos per s | p50 | p95 | Errors |
| --- | --- | --- | --- | --- | --- |
| 1 worker × 2 ONNX threads | 10 | 5.8 | 680 ms | 1.1 s | 0 |
| 2 workers × 1 ONNX thread | 10 | 5.9 | 650 ms | 1.0 s | 0 |
| 2 workers × 1 ONNX thread | 5 | 3.7 | 320 ms | 480 ms | 0 |

- **2 cores handle about 6 photos per second.** 10 cameras ask for about 10, so photos queue and
  the median wait doubles. With 5 cameras every photo is served, with a p95 under half a second.
- **2 workers with 1 thread each** is slightly faster here, but each worker holds its own copy of
  the model: inside the 1 GB container it peaks at 1021 MB. The Docker image therefore runs
  **1 worker with 2 threads**, as fast there (5.4 photos per second) with a 690 MB peak. Workers ×
  threads stays at the number of cores, so they never fight over them.
- A request includes the upload, decoding a 1024 px photo, preprocessing and the model. Intel Core
  Ultra 5 125H, on mains power. Full numbers: [`reports/load/`](reports/load).

### In Docker

The API image ([`docker/api.Dockerfile`](docker/api.Dockerfile)) is built in two stages: one
installs the dependencies with uv, the other keeps only them and one category's ONNX model.

| API image | |
| --- | --- |
| Size | **432 MB** on disk, about 200 MB to download (Python 133 MB, libraries 214 MB, model 109 MB) |
| PyTorch | none: only ONNX Runtime, numpy and Pillow |
| User | `app`, uid 10001, not root |
| Health check | healthy once `/health/ready` answers, that is once the model is loaded |

```bash
make api-up        # builds it and starts it with 2 CPUs and 1 GB; demo page at http://localhost:8000
make smoke         # ready? a good bottle passes? a broken one is rejected?

# the proofs behind the table
docker run --rm --entrypoint python anomaly-inspector-api -c \
  "import sys, anomaly_inspector.serving.app; assert 'torch' not in sys.modules"
docker run --rm --entrypoint id anomaly-inspector-api      # uid=10001(app)
```

`docker image ls` shows 660 MB: Docker's image store counts the unpacked layers and their
compressed copies together. A separate training image ([`docker/train.Dockerfile`](docker/train.Dockerfile),
about 3 GB with CPU PyTorch) holds no code: it runs the pipeline on the repository mounted into it.

### Published on GitHub Container Registry

The API image is public on
[GHCR](https://github.com/users/you-ssef-dan/packages/container/package/anomaly-inspector-api),
GitHub's registry for container images. It runs anywhere with Docker, without the repository or
the data:

```bash
docker run --rm -p 8000:8000 ghcr.io/you-ssef-dan/anomaly-inspector-api:0.1.0
# then open http://localhost:8000 (demo page) or http://localhost:8000/docs
```

![The public anomaly-inspector-api package on GHCR, with its tags 0.1.0 and cc5987c](reports/figures/ghcr_package.png)

| Tag | Means | Used for |
| --- | --- | --- |
| `0.1.0` | the first release (bottle model) | what people and deployments ask for |
| `cc5987c` | the commit it was built from | knowing exactly which code and model are inside |

Both tags name the same image (one digest). There is no `latest` tag: it moves with every build,
so a deployment pinned to it would change without anyone deciding it.

How it was published, once and by hand (CI takes this over in phase 8):

```bash
# 1. give the gh CLI token the right to push packages, and log Docker in to GHCR with it
gh auth refresh -h github.com -s write:packages
gh auth token | docker login ghcr.io -u you-ssef-dan --password-stdin

# 2. build from a clean main and name the image: registry/owner/name:tag
docker compose --profile api build api
SHA=$(git rev-parse --short HEAD)
docker tag anomaly-inspector-api ghcr.io/you-ssef-dan/anomaly-inspector-api:0.1.0
docker tag anomaly-inspector-api ghcr.io/you-ssef-dan/anomaly-inspector-api:$SHA

# 3. push (about 200 MB); the second push only adds a tag, the layers are already there
docker push ghcr.io/you-ssef-dan/anomaly-inspector-api:0.1.0
docker push ghcr.io/you-ssef-dan/anomaly-inspector-api:$SHA
```

4. **Make it public.** A new package starts private, and a private image cannot be pulled without
   logging in. On the package page: *Package settings* → *Change package visibility* → *Public*.
   *Connect repository* links it to this repository, so it shows on the repository page.
5. **Check it as a stranger would**, logged out, with the smoke test (it needs the MVTec bottle
   photos):

   ```bash
   docker logout ghcr.io
   docker run -d --rm -p 8001:8000 --name ghcr-check ghcr.io/you-ssef-dan/anomaly-inspector-api:0.1.0
   scripts/smoke_test.sh http://localhost:8001   # ready, a good bottle passes, a broken one is rejected
   docker stop ghcr-check
   ```

- **Why GHCR:** same GitHub account, the package sits next to the code, and GitHub Actions can push
  to it with its built-in token. A company would more often use its cloud's private registry
  (Artifact Registry, ECR) for access control, vulnerability scans and no pull limits; only the
  host name and the login command change.
- The package page also lists two **untagged** versions. They are part of the image: Docker pushes
  an index that points to the linux/amd64 image and to a build attestation (how it was built).
  Deleting them breaks the image.

### Live on Render

The same image runs on [Render](https://render.com)'s free plan:
**https://anomaly-inspector-5fe2.onrender.com** (the API docs are at `/docs`).

![The live demo rejecting a broken bottle, with the heatmap over the broken area](reports/figures/demo_render.png)

| Render setting | Value | Why |
| --- | --- | --- |
| Source | existing image `ghcr.io/you-ssef-dan/anomaly-inspector-api:0.1.0` | the image tested above, pinned to a tag |
| Instance | Free: 0.1 CPU, 512 MB of RAM, Frankfurt | no cost |
| `PORT` | `8000` | where the app listens (Render expects 10000 by default) |
| `API_THREADS` | `1` | a second ONNX thread does not help on 0.1 CPU |
| Health check path | `/health/ready` | no traffic before the model is loaded |

- **Slow, but enough for a demo.** A photo with its heatmap takes about 3.3 s end to end, 2.5 s of
  it in the model, against 0.2 s on the laptop's 2 cores. The app uses 359 MB of the 512 MB. Both
  were checked before deploying, by running the image locally with the same limits
  (`docker run --cpus 0.1 --memory 512m -e API_THREADS=1 ...`).
- **It sleeps.** After 15 minutes without visits Render stops it, and the next visit waits about a
  minute while it starts again.
- **Checked like the local container**, with the same smoke test:
  `scripts/smoke_test.sh https://anomaly-inspector-5fe2.onrender.com`.
- **Deploys are deliberate.** Render does not watch GHCR: a new version means changing the tag and
  clicking *Manual Deploy*. In phase 8, CI will trigger Render's deploy hook with the new tag, once
  the tests and the smoke test pass.
- **Why Render:** its free plan runs an existing image at no cost. Google Cloud Run would be faster
  (2 CPUs, about 0.2 s per photo, scales to zero) and is what a production deployment would use;
  moving there is one `gcloud run deploy` with the same image.

<details>
<summary><b>Deploy it on Render yourself, step by step</b></summary>

You need a GitHub account. The free plan costs nothing, but Render may ask for a card to verify
the account: it places a temporary $1 authorisation, not a charge.

1. **Sign up** at [dashboard.render.com/register](https://dashboard.render.com/register) with GitHub.
2. **Create the service:** *+ New* → *Web Service*, then under *Source Code* choose *Existing Image*.
3. **Image URL:** `ghcr.io/you-ssef-dan/anomaly-inspector-api:0.1.0`. The image is public, so leave
   the credentials empty and click *Connect*.
4. **Settings:**
   - *Name*: it becomes `<name>.onrender.com`. Render adds a random suffix if another account
     already uses that name, and the subdomain cannot be changed later.
   - *Region*: the closest one (Frankfurt here). *Instance type*: *Free*.
   - *Environment variables*: `PORT` = `8000` and `API_THREADS` = `1`.
   - *Advanced* → *Health Check Path*: `/health/ready`. Leave *Docker Command* empty: the image
     already starts uvicorn.
5. **Click *Deploy web service*.** The *Logs* tab shows `Application startup complete.` and
   `Uvicorn running on http://0.0.0.0:8000`, then *Your service is live*.
6. **Check it:** `scripts/smoke_test.sh https://<name>.onrender.com`, then open the URL on a phone.
7. **Deploy a new version:** *Settings* → *Image URL* → change the tag, then *Manual Deploy* →
   *Deploy latest reference*.

</details>

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
| `make mlflow-up` / `make mlflow-down` | Start or stop MLflow and PostgreSQL (needs Docker and a `.env`) |
| `make api-up` / `make api-down` | Start or stop the API in Docker, with 2 CPUs and 1 GB: http://localhost:8000/docs |
| `make smoke` | Smoke test of the running API: ready, a good part passes, a defective one is rejected |

To train, evaluate and register the models (needs the data, Docker and about 4 GB of free RAM):

```bash
cp .env.example .env    # then put a real password in it
make mlflow-up          # the UI is at http://localhost:5000
uv run --env-file .env dvc repro                                   # every category; re-runs only what changed
uv run --env-file .env python scripts/register_model.py --category screw
```

Without `--env-file`, `dvc repro` writes exactly the same files, just without MLflow runs.

To rebuild the processed data you need the dataset itself. The [data card](docs/data_card.md)
explains where to get it, how it is verified and how it is split.

## How the project is organised

```
src/anomaly_inspector/
  preprocessing.py   one image function shared by training and (later) serving; numpy + Pillow only
  config.py          typed settings read from params.yaml
  tracking.py        what every MLflow run records (code, data and hardware tags, params)
  registry.py        champion and challenger: the promotion rules
  data/              manifest, validation, splits, PyTorch Dataset
  models/            PatchCore, STFPM, backbones
  training/          STFPM training loop (seeded, early stopping, best checkpoint)
  evaluation/        AUROC, AUPRO, thresholds, ROC and PRO curves, failure gallery
  export/            ONNX export with metadata, INT8 quantisation, latency benchmark
  pipelines/         one entry point per DVC stage: prepare, then train, evaluate, export, optimise
  serving/           the FastAPI app, the ONNX inference wrapper, the heatmap (no PyTorch)
scripts/             training with MLflow logging, benchmark vs anomalib, experiments, gallery
docker/              the API, training and MLflow images (docker-compose.yml runs MLflow and the API)
tests/               unit and API tests on generated fake data, slow tests, a Locust load test
notebooks/           exploratory data analysis (no outputs stored in git)
docs/                data card, experiment findings
reports/             benchmark and experiment results, charts, heatmap gallery, failure notes
params.yaml          every tunable value, validated at load time
dvc.yaml             the pipeline, from raw images to a model and its reports per category
```

## Engineering choices

- **No data in git.** Images and models are versioned with [DVC](https://dvc.org); git keeps small
  pointer files. Given the data, a clean clone rebuilds a byte-identical manifest with `dvc repro`.
- **Honest evaluation.** The data is split into `train`, `val_normal`, `calib` and `holdout`.
  Every decision (thresholds, settings) is made on `calib`; the reported numbers come from
  `holdout`. Nothing is tuned on the test set.
- **Every run traceable, promotion by tested code.** Each MLflow run records its commit, data
  hash and settings; a run from uncommitted code can never become the champion.
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
| 3 | Experiment tracking and model registry with MLflow | ✅ |
| 4 | Full DVC pipeline: train, evaluate, export per category | ✅ |
| 5 | ONNX export, optimisation and quantisation | ✅ |
| 6 | FastAPI inference service with pytest | ✅ |
| 7 | Docker and a first deployment | ✅ |
| 8 | CI/CD with GitHub Actions, including model quality gates | ⬜ |
| 9 | Retraining and batch inspection with Airflow | ⬜ |
| 10 | Monitoring and the feedback loop | ⬜ |

## Data and license

The code is [Apache-2.0](LICENSE). The images are **not** part of this repository: MVTec AD is
licensed [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) (non-commercial), so
models trained on it are for learning and portfolio use. If you use the dataset, please cite:

> Paul Bergmann, Michael Fauser, David Sattlegger and Carsten Steger,
> *A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection*, CVPR 2019.
