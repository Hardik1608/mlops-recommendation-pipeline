"""DAG integrity tests. Airflow does not run on Windows, so these skip locally and run in
the dedicated CI job (Linux) where apache-airflow is installed with its constraints file."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("airflow")

DAGS_DIR = Path(__file__).resolve().parents[2] / "dags"


@pytest.fixture(scope="module")
def dagbag():
    sys.path.insert(0, str(DAGS_DIR))
    try:  # Airflow 3.1+
        from airflow.dag_processing.dagbag import DagBag
    except ImportError:  # Airflow 2.x / 3.0
        from airflow.models.dagbag import DagBag
    return DagBag(dag_folder=str(DAGS_DIR), include_examples=False)


def test_dags_import_without_errors(dagbag) -> None:
    assert not dagbag.import_errors, dagbag.import_errors
    assert {"recsys_training", "recsys_drift_monitoring"} <= set(dagbag.dag_ids)


def test_training_dag_structure(dagbag) -> None:
    dag = dagbag.dags["recsys_training"]
    tasks = {t.task_id: t for t in dag.tasks}
    assert set(tasks) == {
        "ingest",
        "validate_and_split",
        "build_features",
        "feast_apply",
        "feast_materialize",
        "train_evaluate_register",
    }
    assert tasks["validate_and_split"].upstream_task_ids == {"ingest"}
    assert tasks["train_evaluate_register"].upstream_task_ids == {"validate_and_split"}
    assert tasks["feast_materialize"].upstream_task_ids == {"feast_apply"}
    assert dag.catchup is False and dag.max_active_runs == 1
    for task in dag.tasks:
        assert task.retries == 2


def test_drift_dag_triggers_training(dagbag) -> None:
    dag = dagbag.dags["recsys_drift_monitoring"]
    tasks = {t.task_id: t for t in dag.tasks}
    assert set(tasks) == {"compute_drift", "drift_detected", "trigger_retraining"}
    assert tasks["trigger_retraining"].trigger_dag_id == "recsys_training"
    assert tasks["trigger_retraining"].upstream_task_ids == {"drift_detected"}
