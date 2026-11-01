# The training image: the `train` extra (CPU PyTorch, DVC, MLflow) and git, but not the code.
#
# The code comes from the repository mounted at /repo, so a run always uses the exact commit that
# is checked out there, never a copy frozen in an old image. Airflow runs it this way in Phase 9:
#   docker build -f docker/train.Dockerfile -t anomaly-inspector-train .
#   docker run --rm --user "$(id -u):$(id -g)" -v "$PWD:/repo" anomaly-inspector-train dvc repro
# uv as a stage of its own: Dependabot only updates images named on FROM lines.
FROM ghcr.io/astral-sh/uv:0.13.5 AS uv

FROM python:3.12-slim-trixie
COPY --from=uv /uv /bin/uv
# git: DVC, and the run tags (git SHA, dirty flag), read the repository through it.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
# The venv lives in /opt, outside /repo, which the mount covers.
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv
WORKDIR /opt
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-dev --extra train --no-install-project
# Not root. The mounted repository belongs to another user, which git refuses unless told.
RUN useradd --create-home --uid 10001 app \
    && git config --system --add safe.directory /repo
# PYTHONPATH: the package is imported from the mounted src/. HOME in /tmp: any --user can write
# its caches there. USER: DVC asks for a user name, and a --user id unknown to this image has none.
ENV PATH="/opt/venv/bin:$PATH" PYTHONPATH=/repo/src PYTHONUNBUFFERED=1 HOME=/tmp USER=app
USER app
WORKDIR /repo
CMD ["dvc", "repro"]
