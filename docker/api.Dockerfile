# The API image: FastAPI + ONNX Runtime + one model. No PyTorch, no compilers, not root.
#
#   docker build -f docker/api.Dockerfile --build-arg MODEL_FILE=models/patchcore/screw/model.onnx \
#       -t anomaly-inspector-api .
#   docker run --rm -p 8000:8000 anomaly-inspector-api

# uv as a stage of its own: Dependabot only updates images named on FROM lines.
FROM ghcr.io/astral-sh/uv:0.13.5 AS uv

# Stage 1: build the virtual environment with uv. Nothing of this stage but .venv is kept.
FROM python:3.12-slim-trixie AS builder
COPY --from=uv /uv /bin/uv
# Compile .pyc now (faster start), copy files instead of linking from the cache, and use the
# image's Python rather than downloading one.
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
# Dependencies first, in their own layer: it is rebuilt only when uv.lock or pyproject.toml change,
# not on every code edit. Only the `serve` extra: no torch, no training libraries.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-dev --extra serve --no-install-project
# Then the project itself, installed as a normal package (not editable), so src/ is not needed
# in the final image.
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --extra serve --no-editable

# Stage 2: the image that runs. Same base, so the virtual environment's Python links still work.
FROM python:3.12-slim-trixie
ARG MODEL_FILE=models/patchcore/bottle/model.onnx
# The commit the image is built from: /health/live returns it (CD passes it, `dev` otherwise).
ARG BUILD=dev
# A fixed, unprivileged user: a bug in the app cannot touch the rest of the container as root.
RUN useradd --create-home --uid 10001 app
WORKDIR /app
COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --chown=app:app ${MODEL_FILE} /app/model/model.onnx
# 1 worker x 2 ONNX Runtime threads, for 2 CPUs and 1 GB. Measured under load in this container:
# as fast as 2 workers x 1 thread (5.4 photos/s), but a peak of 690 MB instead of 1021 MB, since
# each worker holds its own copy of the model. Both can be changed at `docker run -e`.
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1 \
    API_MODEL_PATH=/app/model/model.onnx API_THREADS=2 UVICORN_WORKERS=1 API_BUILD=${BUILD}
USER app
EXPOSE 8000
# Docker marks the container healthy only once the model is loaded (/health/ready answers 200).
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s CMD \
    python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/ready')"
CMD ["uvicorn", "anomaly_inspector.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
