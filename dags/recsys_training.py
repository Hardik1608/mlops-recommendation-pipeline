"""Daily batch pipeline for the recommendation system.

ingest -> validate + temporal split -> (batch features -> feast apply -> materialize)
                                    -> (train -> evaluate vs baseline -> register -> gate)

The gate never promotes silently: a model that beats the champion becomes the
``challenger`` alias (A/B traffic); promotion to ``champion`` is an explicit step.
"""

from __future__ import annotations

from datetime import datetime

from recsys_common import DATASET, DEFAULT_ARGS, recsys_task

try:  # Airflow 3.x
    from airflow.sdk import dag
except ImportError:  # Airflow 2.x
    from airflow.decorators import dag


@dag(
    dag_id="recsys_training",
    schedule="@daily",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    tags=["recsys", "training"],
    doc_md=__doc__,
)
def recsys_training():
    @recsys_task()
    def ingest(dataset: str) -> str:
        from recsys.pipelines.tasks import ingest as run

        return run(dataset)

    @recsys_task()
    def validate_and_split(dataset: str) -> dict:
        from recsys.pipelines.tasks import validate_and_split as run

        return run(dataset)

    @recsys_task()
    def build_features(dataset: str) -> dict:
        from recsys.pipelines.tasks import build_features as run

        return run(dataset)

    @recsys_task()
    def feast_apply() -> dict:
        from recsys.pipelines.tasks import feast_apply as run

        return run()

    @recsys_task()
    def feast_materialize() -> dict:
        from recsys.pipelines.tasks import feast_materialize as run

        return run()

    @recsys_task()
    def train_evaluate_register(dataset: str) -> dict:
        from recsys.pipelines.tasks import train as run

        summary = run(dataset)
        # keep the XCom small: decision + headline metrics only
        return {
            "model_version": summary["model_version"],
            "decision": summary["decision"],
            "ndcg_at_10": summary["metrics"].get("ndcg_at_10"),
        }

    raw = ingest(DATASET)
    split = validate_and_split(DATASET)
    features = build_features(DATASET)
    trained = train_evaluate_register(DATASET)
    raw >> split >> [features, trained]
    features >> feast_apply() >> feast_materialize()


recsys_training()
