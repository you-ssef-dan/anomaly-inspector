.PHONY: setup lint test test-slow bench check mlflow-up mlflow-down
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
