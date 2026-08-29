# syntax=docker/dockerfile:1

# uv binary, pinned to the version the lockfile was made with.
FROM ghcr.io/astral-sh/uv:0.9.5 AS uv

# Builder: resolve and install the locked dependencies into /app/.venv.
FROM python:3.12-slim-bookworm AS builder
COPY --from=uv /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Runtime: the virtual environment and the code, nothing else.
FROM python:3.12-slim-bookworm AS runtime
RUN useradd --create-home --uid 1000 lm
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY --chown=lm:lm lm ./lm
COPY --chown=lm:lm configs ./configs
COPY --chown=lm:lm data ./data
COPY --chown=lm:lm docs ./docs
RUN mkdir -p /app/runs && chown lm:lm /app /app/runs

# One thread and float64 keep every run exactly repeatable.
ENV PATH=/app/.venv/bin:$PATH \
    PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

USER lm
ENTRYPOINT ["python", "-m"]
CMD ["lm.train", "--help"]
