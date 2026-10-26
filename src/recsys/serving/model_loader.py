"""Load champion/challenger/shadow packages from local directories or MLflow aliases."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from recsys.config import ModelSource, Settings
from recsys.models.package import ModelPackage, load_package
from recsys.serving.ab import CHALLENGER, CHAMPION, SHADOW

logger = logging.getLogger(__name__)


@dataclass
class LoadedModel:
    variant: str
    package: ModelPackage
    version: str
    source: str
    loaded_at: float = field(default_factory=time.time)

    @property
    def model_type(self) -> str:
        return self.package.model.model_type


def load_local(path: Path, variant: str) -> LoadedModel:
    path = Path(path)
    package = load_package(path)
    version_file = path / "REGISTRY_VERSION"
    version = (
        version_file.read_text(encoding="utf-8").strip()
        if version_file.exists()
        else package.package_id
    )
    return LoadedModel(variant=variant, package=package, version=version, source=str(path))


def resolve_mlflow_versions(settings: Settings) -> dict[str, str]:
    """Current alias -> version mapping (used for hot-swap polling)."""
    import mlflow

    from recsys.training.registry import ModelRegistry

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    registry = ModelRegistry(settings.registered_model_name)
    out: dict[str, str] = {}
    for alias in (CHAMPION, CHALLENGER, SHADOW):
        version = registry.version_for(alias)
        if version is not None:
            out[alias] = version
    return out


def load_mlflow(settings: Settings, variant: str, version: str) -> LoadedModel:
    import mlflow
    import mlflow.pyfunc

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    uri = f"models:/{settings.registered_model_name}/{version}"
    pyfunc_model = mlflow.pyfunc.load_model(uri)
    package = pyfunc_model.unwrap_python_model().package
    return LoadedModel(variant=variant, package=package, version=str(version), source=uri)


def load_models(settings: Settings) -> dict[str, LoadedModel]:
    """Champion is mandatory; challenger (A/B split) and shadow are optional."""
    models: dict[str, LoadedModel] = {}
    if settings.model_source == ModelSource.MLFLOW:
        for variant, version in resolve_mlflow_versions(settings).items():
            models[variant] = load_mlflow(settings, variant, version)
    else:
        models[CHAMPION] = load_local(settings.champion_model_dir, CHAMPION)
        for variant, directory in (
            (CHALLENGER, settings.challenger_model_dir),
            (SHADOW, settings.shadow_model_dir),
        ):
            if directory is not None and Path(directory, "package.json").exists():
                models[variant] = load_local(directory, variant)
    if CHAMPION not in models:
        raise RuntimeError("no champion model available (train one with `recsys train`)")
    for variant, loaded in models.items():
        logger.info(
            "loaded %s: %s v%s from %s", variant, loaded.model_type, loaded.version, loaded.source
        )
    return models
