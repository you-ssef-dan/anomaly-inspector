# Phase 8: CI/CD with GitHub Actions

**Week 6** · [Full details, code and pitfalls](../PROJECT_GUIDE.md#phase-8-cicd-with-github-actions)

**Goal:** No change reaches production unless the code passes its tests, the model passes its quality gates, and the image passes a smoke test and a vulnerability scan.

## Steps

- [ ] **1.** Add a `docker` job to `ci.yml`: build with a tiny model, run the smoke test, scan the image.
- [ ] **2.** Write `src/anomaly_inspector/pipelines/gate.py` and a `gates` section in `params.yaml` (floors about one point below your Phase 2 baseline), with tests.
- [ ] **3.** Add `.github/workflows/model-ci.yml`:
  - [ ] Switch the DVC remote to Cloudflare R2 (commands in Phase 1, step 3) and add `R2_ACCESS_KEY_ID` and `R2_SECRET_ACCESS_KEY` as GitHub secrets
  - [ ] Workflow: `dvc repro --pull export`, the metrics diff as a pull request comment, then the gate
- [ ] **4.** Add `.github/workflows/cd.yml`:
  - [ ] Create a `production` environment with yourself as required reviewer
  - [ ] Set up Google Cloud OIDC (Workload Identity Federation, restricted to your repository) and add `GCP_WIF_PROVIDER` and `GCP_DEPLOY_SA` as repository variables
  - [ ] Workflow: pull the real model with DVC, build and push to GHCR, deploy to Cloud Run, run the live smoke test
- [ ] **5.** Protect `main`: require `checks`, `docker` and `pipeline`, and know the rollback (`gcloud run services update-traffic`).
- [ ] **6.** Harden the workflows:
  - [ ] Pin every action to a full commit SHA (pinact)
  - [ ] Declare `permissions:` in every workflow (least privilege)
  - [ ] Never use `pull_request_target` with a checkout of pull request code
  - [ ] No long-lived cloud keys: OIDC for Google Cloud, a bucket-scoped token for R2
  - [ ] Dependabot for `github-actions`, `uv` and `docker`, weekly

## Done when

- [ ] Every pull request shows green `checks` and `docker`, plus `pipeline` when ML code changed.
- [ ] A deliberately bad pull request (coreset ratio 0.0001) gets a metrics comment and a red gate. Close it unmerged.
- [ ] A merge to main deploys a new revision after your approval, and the live smoke test passes.
- [ ] Every action is pinned by SHA, and Dependabot is on.
- [ ] The README shows the three workflow badges.

---

[← Phase 7: Docker and the first deployment](07-docker-and-first-deployment.md) · [All phases](README.md) · [Phase 9: Orchestration with Airflow 3 →](09-airflow-orchestration.md)
