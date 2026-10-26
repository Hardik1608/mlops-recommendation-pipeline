"""Hourly drift monitoring: PSI of recent streamed events vs the training window.

The report is pushed to the Prometheus Pushgateway (Grafana panel + alert) and, when any
distribution drifts beyond the threshold (PSI > 0.2), the training DAG is triggered.
"""

from __future__ import annotations

from datetime import datetime

from recsys_common import DATASET, DEFAULT_ARGS, recsys_task

try:  # Airflow 3.x
    from airflow.sdk import dag, task
except ImportError:  # Airflow 2.x
    from airflow.decorators import dag, task

try:  # Airflow 3.x (standard provider)
    from airflow.providers.standard.operators.trigger_dagrun import TriggerDagRunOperator
except ImportError:  # Airflow 2.x
    from airflow.operators.trigger_dagrun import TriggerDagRunOperator


@dag(
    dag_id="recsys_drift_monitoring",
    schedule="@hourly",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    tags=["recsys", "monitoring"],
    doc_md=__doc__,
)
def recsys_drift_monitoring():
    @recsys_task()
    def compute_drift(dataset: str) -> dict:
        from recsys.pipelines.tasks import drift

        report = drift(dataset)
        return {"drift_detected": report["drift_detected"], "features": report["features"]}

    @task.short_circuit
    def drift_detected(report: dict) -> bool:
        return bool(report.get("drift_detected"))

    retrain = TriggerDagRunOperator(
        task_id="trigger_retraining",
        trigger_dag_id="recsys_training",
        wait_for_completion=False,
    )
    drift_detected(compute_drift(DATASET)) >> retrain


recsys_drift_monitoring()
