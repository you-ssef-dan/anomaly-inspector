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
