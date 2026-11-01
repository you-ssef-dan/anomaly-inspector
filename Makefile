.PHONY: setup lint test test-slow bench check mlflow-up mlflow-down api-up api-down airflow-up airflow-down smoke
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
# needs about 3 GB of RAM for bottle and more for carpet and screw.
bench:
	for c in bottle carpet screw; do \
		for impl in ours anomalib-v1; do \
			uv run python scripts/benchmark_anomalib.py --category $$c --impl $$impl || exit 1; \
		done; \
	done
check: lint test
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
# Airflow, after MLflow and the API: they create the `inspection` network it joins.
# Needs airflow/.env (cp airflow/.env.example airflow/.env, then fill it). UI: http://localhost:8080
airflow-up: mlflow-up api-up
	cd airflow && docker compose up -d
airflow-down:
	cd airflow && docker compose down
# Against the running API: ready, a good bottle passes, a broken one is rejected.
smoke:
	scripts/smoke_test.sh
