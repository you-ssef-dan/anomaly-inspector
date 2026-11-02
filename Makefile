.PHONY: setup lint test test-slow bench check mlflow-up mlflow-down api-up api-down monitoring-up monitoring-down test-alerts airflow-up airflow-down train-image smoke test-dags
setup:
	uv sync --all-extras
	uv run pre-commit install
lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy src
test:
	uv run pytest -m "not slow and not integration"
test-slow:
	uv run pytest -m slow --no-cov
# One process per run releases the feature bank in between. Close other apps first: each run
# needs about 3 GB of RAM for bottle and more for carpet and screw. The PatchCores take a few
# minutes per category, STFPM trains for 20 to 40. Some only: make bench BENCH_IMPLS="ours stfpm"
BENCH_IMPLS ?= ours anomalib-v1 stfpm
bench:
	for c in bottle carpet screw; do \
		for impl in $(BENCH_IMPLS); do \
			uv run python scripts/benchmark_anomalib.py --category $$c --impl $$impl || exit 1; \
		done; \
	done
check: lint test
# The DAG tests run in Airflow's own image, like the DAGs (CI's airflow-dags job runs this). The
# project's pytest settings and tests/conftest.py need the project's environment: left out.
test-dags:
	docker run --rm -v "$(CURDIR):/repo:ro" -w /repo apache/airflow:3.3.2 bash -c \
		"pip install --quiet --disable-pip-version-check pytest==9.1.1 && \
		python -m pytest -p no:cacheprovider -o addopts= --confcutdir=tests/airflow tests/airflow"
# Needs .env (cp .env.example .env). Stopping keeps the runs: they live in Docker volumes.
mlflow-up:
	docker compose --profile mlflow up -d --build
mlflow-down:
	docker compose --profile mlflow down
# The API in Docker, with 2 CPUs and 1 GB like a small cloud instance: http://localhost:8000/docs
api-up:
	docker compose --profile api up -d --build api
api-down:
	docker compose --profile api rm --stop --force api
# Prometheus (http://localhost:9090) and Grafana (http://localhost:3000), with the API they watch.
# Stopping keeps Prometheus's history: it lives in a Docker volume. Both are re-created each time,
# so they read monitoring/ again: after a change there, run this. A running container also keeps
# seeing the old folder when git deletes and re-creates it (switching to a branch without it).
monitoring-up: api-up
	docker compose --profile monitoring up -d --force-recreate
monitoring-down:
	docker compose --profile monitoring rm --stop --force prometheus grafana
# Prometheus's own checker on its config and alert rules, then the rules' tests (CI's alert-rules
# job runs this). The image is the one in docker-compose.yml.
test-alerts:
	docker run --rm -v "$(CURDIR)/monitoring/prometheus:/etc/prometheus:ro" -w /etc/prometheus \
		--entrypoint sh prom/prometheus:v3.15.1 -c \
		"promtool check config prometheus.yml && promtool test rules alerts.test.yml"
# Airflow, after MLflow and the API: they create the `inspection` network it joins.
# Needs airflow/.env (cp airflow/.env.example airflow/.env, then fill it). UI: http://localhost:8080
# The DagsHub token comes from .dvc/config.local, its only copy on this laptop.
airflow-up: mlflow-up api-up
	token="$$(uv run dvc config --local remote.dagshub.secret_access_key)" && \
		cd airflow && DAGSHUB_TOKEN="$$token" docker compose up -d
airflow-down:
	cd airflow && docker compose down
# The image Airflow trains in. Rebuild it when uv.lock changes: the DAG refuses a stale one.
train-image:
	docker build -f docker/train.Dockerfile -t anomaly-inspector-train .
# Against the running API: ready, a good bottle passes, a broken one is rejected.
smoke:
	scripts/smoke_test.sh
