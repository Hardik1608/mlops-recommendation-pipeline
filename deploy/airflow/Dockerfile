# Airflow image for the optional `airflow` compose profile.
# The project and its ML dependencies (MLflow, Feast, ...) are installed into a SEPARATE
# virtualenv so they can never conflict with Airflow's pinned constraints; DAG tasks run
# there through @task.external_python (see dags/recsys_common.py). On Kubernetes the
# equivalent is KubernetesPodOperator running the recsys image.
FROM apache/airflow:3.3.2-python3.11

USER root
RUN python -m venv /opt/recsys-venv \
    && mkdir -p /app/data /app/artifacts /app/reports \
    && chown -R airflow:root /opt/recsys-venv /app
USER airflow

COPY --chown=airflow:root requirements.txt /tmp/requirements.txt
RUN /opt/recsys-venv/bin/pip install --no-cache-dir -r /tmp/requirements.txt
COPY --chown=airflow:root pyproject.toml README.md /opt/recsys-src/
COPY --chown=airflow:root src /opt/recsys-src/src
RUN /opt/recsys-venv/bin/pip install --no-cache-dir --no-deps /opt/recsys-src
COPY --chown=airflow:root configs /app/configs
COPY --chown=airflow:root feature_repo /app/feature_repo

ENV RECSYS_PYTHON=/opt/recsys-venv/bin/python \
    RECSYS_DATA_DIR=/app/data \
    RECSYS_ARTIFACTS_DIR=/app/artifacts \
    RECSYS_REPORTS_DIR=/app/reports \
    RECSYS_CONFIGS_DIR=/app/configs \
    RECSYS_FEAST_REPO_PATH=/app/feature_repo
