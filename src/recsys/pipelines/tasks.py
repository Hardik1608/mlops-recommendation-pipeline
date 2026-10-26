"""Pipeline steps as plain functions with JSON-serialisable return values.

Airflow DAGs stay thin: each task calls one function here, so all logic is unit-testable
without an Airflow installation, and the same steps run from the CLI or a notebook.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from recsys.config import Settings, load_train_config

logger = logging.getLogger(__name__)


def _settings(settings: Settings | None) -> Settings:
    return settings or Settings()


def ingest(dataset: str | None = None, settings: Settings | None = None) -> str:
    from recsys.data.prepare import ensure_raw

    s = _settings(settings)
    return str(ensure_raw(s, dataset or s.dataset))


def validate_and_split(dataset: str | None = None, settings: Settings | None = None) -> dict:
    from recsys.data.prepare import prepare_dataset

    s = _settings(settings)
    dataset = dataset or s.dataset
    return prepare_dataset(s, dataset, load_train_config(s, dataset).split).info


def build_features(dataset: str | None = None, settings: Settings | None = None) -> dict:
    from recsys.data.prepare import load_prepared
    from recsys.features.offline import write_feature_tables

    s = _settings(settings)
    prepared = load_prepared(s, dataset or s.dataset)
    paths = write_feature_tables(prepared.split.train_val, prepared.movies, s.features_dir)
    return {name: str(path) for name, path in paths.items()}


def feast_apply(settings: Settings | None = None) -> dict:
    """``feast apply`` for the configured repo/yaml (registers entities, views, services)."""
    s = _settings(settings)
    repo = Path(s.feast_repo_path).resolve()
    cmd = [
        sys.executable,
        "-c",
        "from feast.cli.cli import cli; cli()",
        "-c",
        str(repo),
        "-f",
        str(repo / s.feast_yaml),
        "apply",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"feast apply failed:\n{proc.stdout}\n{proc.stderr}")
    logger.info("feast apply: %s", proc.stdout.strip().splitlines()[-1:] or "ok")
    return {"repo": str(repo), "yaml": s.feast_yaml, "output": proc.stdout.strip()[-2000:]}


def feast_materialize(settings: Settings | None = None) -> dict:
    """Load the batch feature views into the online store (sqlite locally, Redis in docker)."""
    from feast import FeatureStore

    s = _settings(settings)
    repo = Path(s.feast_repo_path).resolve()
    store = FeatureStore(repo_path=str(repo), fs_yaml_file=repo / s.feast_yaml)
    views = ["user_stats", "item_stats"]
    store.materialize(
        start_date=datetime(1990, 1, 1, tzinfo=UTC),
        end_date=datetime.now(tz=UTC),
        feature_views=views,
    )
    return {"materialized": views}


def train(dataset: str | None = None, settings: Settings | None = None) -> dict[str, Any]:
    from recsys.training.train import summarize, train_and_register

    s = _settings(settings)
    return summarize(train_and_register(s, dataset or s.dataset))


def drift(dataset: str | None = None, settings: Settings | None = None) -> dict[str, Any]:
    from recsys.monitoring.drift import run_drift_job

    s = _settings(settings)
    return run_drift_job(s, dataset or s.dataset)


def should_retrain(report: dict[str, Any]) -> bool:
    """Retraining trigger used by the drift DAG."""
    return bool(report.get("drift_detected"))
