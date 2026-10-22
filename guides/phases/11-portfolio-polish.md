# Portfolio polish

**Week 8, days 4 and 5** · [Full details, code and pitfalls](../PROJECT_GUIDE.md#portfolio-polish)

**Goal:** A recruiter understands the project in 30 seconds, and an engineer can run it in 10 minutes. Write the README, model card and decision records in your own words; interviewers will ask about every decision, and writing them yourself is how you get ready.

## Steps

- [ ] **README**, in this order: pitch and badges, demo GIF and live link, results table, architecture diagram, quickstart, how it works, design decisions, limitations, data and license.
- [ ] **Model card** in `docs/model_card.md` (sections from *Model Cards for Model Reporting*), next to the data card from Phase 1.
- [ ] **Decision records** in `docs/adr/`, each with context, decision and consequences:
  - [ ] PatchCore in production, STFPM as the fast alternative, with your numbers
  - [ ] ONNX Runtime for serving, no PyTorch in the image
  - [ ] DVC for data and pipeline, MLflow for runs and the registry
  - [ ] A threshold calibrated on held-out data, with a stated 10:1 cost ratio
  - [ ] Airflow proposes, CI decides: retraining reaches production only through a pull request
- [ ] **CV bullets** with numbers you measured (templates in the guide):
  - [ ] PatchCore in PyTorch: image AUROC and AUPRO on the holdout split
  - [ ] FastAPI and ONNX Runtime: image size, and INT8 latency change
  - [ ] DVC, MLflow registry, Airflow and GitHub Actions gates
  - [ ] Monitoring: time to detect the simulated lighting failure
- [ ] **Get it seen:**
  - [ ] Post a short LinkedIn update each week, each with one picture
  - [ ] Publish one longer write-up (demo GIF, failure gallery, drift test) and link it from the README
  - [ ] Record a 2-minute walkthrough video: demo, architecture, one failure case, one design decision
  - [ ] Pin the repo on your GitHub profile

## Done when

- [ ] A friend who has never seen the repo runs the quickstart in under 10 minutes.
- [ ] The README shows the demo, live link, results table, diagram and badges at the top.
- [ ] The model card, data card and five decision records are written in your own words.
- [ ] gitleaks finds no secrets in the history, and the code has a LICENSE file.
- [ ] Your CV and LinkedIn show the project with your real numbers.

---

[← Phase 10: Monitoring and the feedback loop](10-monitoring-and-feedback-loop.md) · [All phases](README.md)
