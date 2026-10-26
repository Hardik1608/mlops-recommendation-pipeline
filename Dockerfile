# syntax=docker/dockerfile:1.7
# One image for every Python workload (API, stream consumer, producer, training jobs);
# the command differs per container. Multi-stage: build deps in a venv, copy it into a
# slim runtime that runs as a non-root user.

FROM python:3.11-slim-bookworm AS builder
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
WORKDIR /build
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps .

FROM python:3.11-slim-bookworm AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:${PATH}" \
    MLFLOW_DISABLE_AGENT_HINT=1 \
    RECSYS_LOG_FORMAT=json
RUN useradd --create-home --uid 10001 app
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY configs ./configs
COPY feature_repo ./feature_repo
RUN mkdir -p data artifacts reports && chown -R app:app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=60s --retries=3 \
  CMD python -c "import sys, urllib.request; sys.exit(urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).status != 200)"
CMD ["uvicorn", "recsys.serving.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
