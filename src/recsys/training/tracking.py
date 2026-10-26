"""MLflow setup shared by training, tuning and benchmarking."""

from __future__ import annotations

from pathlib import Path

from recsys.config import Settings


def _is_local(uri: str) -> bool:
    return uri.startswith(("sqlite:", "file:")) or "://" not in uri


def configure_mlflow(settings: Settings, experiment: str | None = None) -> str:
    """Point MLflow at the configured backend and return the experiment id."""
    import mlflow

    uri = settings.mlflow_tracking_uri
    if uri.startswith("sqlite:///"):
        Path(uri.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(uri)

    name = experiment or settings.mlflow_experiment
    existing = mlflow.get_experiment_by_name(name)
    if existing is not None:
        mlflow.set_experiment(name)
        return existing.experiment_id

    artifact_location = settings.mlflow_artifact_root
    if artifact_location is None and _is_local(uri):
        root = Path("mlruns/artifacts").resolve()
        root.mkdir(parents=True, exist_ok=True)
        artifact_location = root.as_uri()
    experiment_id = mlflow.create_experiment(name, artifact_location=artifact_location)
    mlflow.set_experiment(name)
    return experiment_id
