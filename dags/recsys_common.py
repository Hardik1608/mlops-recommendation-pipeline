"""Shared helpers for the recsys DAGs (no DAG is defined here).

Every task body imports ``recsys.pipelines.tasks`` lazily, so the DAG files parse with only
Airflow installed (scheduler/CI) and the heavy ML stack lives in the task runtime.
"""

from __future__ import annotations

import os
from datetime import timedelta

try:  # Airflow 3.x
    from airflow.sdk import task
except ImportError:  # Airflow 2.x
    from airflow.decorators import task

RECSYS_PYTHON = os.environ.get("RECSYS_PYTHON", "")
DATASET = os.environ.get("RECSYS_DATASET", "ml-latest-small")

DEFAULT_ARGS = {
    "owner": "ml-platform",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "retry_exponential_backoff": True,
}


def recsys_task(**kwargs):
    """``@task`` that runs inside the project's own virtualenv when it exists.

    Isolating the project's dependencies (MLflow, Feast, scipy...) from Airflow's own
    pinned constraints avoids the classic "pip install into the scheduler" breakage.
    """
    if RECSYS_PYTHON and os.path.exists(RECSYS_PYTHON):
        return task.external_python(python=RECSYS_PYTHON, expect_airflow=False, **kwargs)
    return task(**kwargs)
