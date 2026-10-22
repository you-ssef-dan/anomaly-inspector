# Phase 0: Repo foundations

**Week 1, days 1 and 2** · [Full details, code and pitfalls](../PROJECT_GUIDE.md#phase-0-repo-foundations)

**Goal:** A professional skeleton. One command installs everything, one command runs every check, and CI runs the same checks on each pull request.

## Steps

- [ ] **1.** Install uv and create the repo at `~/projects/industrial-anomaly-detection`, not inside this folder (its path has spaces and an `&`).
- [ ] **2.** Create the folder layout: `src/anomaly_inspector/`, `tests/`, `docker/`, `airflow/dags/`, `docs/`, `monitoring/`, `notebooks/` and `.github/workflows/`.
- [ ] **3.** Point uv at CPU-only PyTorch wheels in `pyproject.toml`, before you add torch.
- [ ] **4.** Add dependencies in groups (`train` and `serve` extras, plus dev tools) so the serving image never installs PyTorch.
- [ ] **5.** Configure ruff, mypy and pytest in `pyproject.toml`.
- [ ] **6.** Add `.pre-commit-config.yaml`, then run `uv run pre-commit install`.
- [ ] **7.** Add a `Makefile` with `setup`, `lint`, `test` and `check` (recipe lines start with a tab).
- [ ] **8.** Add the first CI workflow, `.github/workflows/ci.yml`, with every action pinned to a commit SHA.
- [ ] **9.** Create `params.yaml` and the Pydantic models in `src/anomaly_inspector/config.py`.
- [ ] **10.** Set up the git workflow on GitHub: protect `main` (pull request and green CI required), one issue per phase, Conventional Commits, squash-merge.

## Done when

- [ ] A fresh clone runs `make setup && make check` with no errors.
- [ ] pre-commit blocks a commit that has a lint error.
- [ ] CI runs on a pull request, passes, and branch protection requires it.
- [ ] `uv run python -c "import torch; print(torch.__version__)"` prints a version ending in +cpu.
- [ ] README has a one-line pitch, the CI badge and setup steps.

---

[All phases](README.md) · [Phase 1: Data and DVC →](01-data-and-dvc.md)
