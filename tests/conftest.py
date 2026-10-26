"""Shared fixtures: a synthetic MovieLens-shaped dataset prepared in a temp directory.

Everything runs offline (no downloads, no brokers, no Docker) so CI is hermetic.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest

from recsys.config import Settings, TrainConfig, load_train_config
from recsys.data.prepare import PreparedData, prepare_dataset
from recsys.data.synthetic import write_synthetic
from recsys.models import ImplicitALS, PopularityRecommender

REPO_ROOT = Path(__file__).resolve().parents[1]


def make_settings(root: Path, **overrides: object) -> Settings:
    root.mkdir(parents=True, exist_ok=True)
    configs = root / "configs"
    if not configs.exists():
        shutil.copytree(REPO_ROOT / "configs", configs)
    values: dict[str, object] = {
        "data_dir": root / "data",
        "artifacts_dir": root / "artifacts",
        "reports_dir": root / "reports",
        "configs_dir": configs,
        "dataset": "synthetic",
        "mlflow_tracking_uri": f"sqlite:///{(root / 'mlflow.db').as_posix()}",
        "mlflow_artifact_root": (root / "mlartifacts").resolve().as_uri(),
        "champion_model_dir": root / "artifacts" / "models" / "champion",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)  # type: ignore[arg-type]


@pytest.fixture(scope="session")
def session_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("recsys")


@pytest.fixture(scope="session")
def settings(session_root: Path) -> Settings:
    s = make_settings(session_root)
    write_synthetic(s.raw_dir / "synthetic")
    return s


@pytest.fixture(scope="session")
def train_config(settings: Settings) -> TrainConfig:
    return load_train_config(settings, "synthetic")


@pytest.fixture(scope="session")
def prepared(settings: Settings, train_config: TrainConfig) -> PreparedData:
    return prepare_dataset(settings, "synthetic", train_config.split)


@pytest.fixture(scope="session")
def ratings(prepared: PreparedData) -> pd.DataFrame:
    split = prepared.split
    return pd.concat([split.train, split.val, split.test], ignore_index=True)


@pytest.fixture(scope="session")
def implicit_model(prepared: PreparedData) -> ImplicitALS:
    return ImplicitALS(factors=16, regularization=1.0, alpha=5.0, iterations=8).fit(
        prepared.split.train_val
    )


@pytest.fixture(scope="session")
def popularity_model(prepared: PreparedData) -> PopularityRecommender:
    return PopularityRecommender().fit(prepared.split.train_val)
