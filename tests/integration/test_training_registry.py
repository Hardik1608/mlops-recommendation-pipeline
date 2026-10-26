"""End-to-end: train, MLflow run, registry, gate, promote, rollback, serve from alias."""

from __future__ import annotations

import pandas as pd
import pytest

from recsys.config import ModelSource
from recsys.serving.model_loader import load_models
from recsys.training.registry import CHALLENGER, CHAMPION, PREVIOUS, SHADOW, ModelRegistry
from recsys.training.train import train_and_register

pytestmark = [pytest.mark.integration, pytest.mark.slow]


@pytest.fixture(scope="module")
def runs(settings, train_config, prepared):
    import mlflow

    weak = train_config.model_copy(
        update={"model": train_config.model.model_copy(update={"factors": 2, "iterations": 1})}
    )
    first = train_and_register(settings, "synthetic", config=weak)
    second = train_and_register(settings, "synthetic", config=train_config)
    third = train_and_register(settings, "synthetic", config=weak)
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    return first, second, third


def test_first_model_bootstraps_champion(runs, settings) -> None:
    first, _, _ = runs
    assert first.decision.action == "bootstrap_champion"
    assert first.model_version == "1"
    assert first.metrics["ndcg_at_10"] > 0
    assert "ndcg_at_10" in first.baseline_metrics


def test_better_candidate_becomes_challenger_worse_is_rejected(runs) -> None:
    first, second, third = runs
    assert second.decision.action == "challenger"
    assert second.decision.champion_value == pytest.approx(first.metrics["ndcg_at_10"])
    assert third.decision.action == "rejected"


def test_aliases_and_local_export(runs, settings) -> None:
    registry = ModelRegistry(settings.registered_model_name)
    assert registry.aliases() == {CHAMPION: "1", CHALLENGER: "2", PREVIOUS: None, SHADOW: None}
    champion_dir = settings.artifacts_dir / "models" / "champion"
    assert (champion_dir / "REGISTRY_VERSION").read_text() == "1"
    assert (settings.artifacts_dir / "models" / "challenger" / "package.json").exists()


def test_mlflow_run_has_params_metrics_and_gate_tags(runs, settings) -> None:
    from mlflow import MlflowClient

    run = MlflowClient().get_run(runs[1].run_id)
    assert run.data.params["dataset"] == "synthetic"
    assert "model.factors" in run.data.params
    assert "baseline_ndcg_at_10" in run.data.metrics
    assert run.data.tags["gate_action"] == "challenger"


def test_pyfunc_batch_scoring(runs, settings) -> None:
    import mlflow.pyfunc

    model = mlflow.pyfunc.load_model(f"models:/{settings.registered_model_name}@{CHAMPION}")
    out = model.predict(pd.DataFrame({"user_id": [1, 2, 10**9]}), params={"k": 3})
    assert len(out) == 9
    assert set(out.columns) >= {"user_id", "rank", "item_id", "score", "strategy"}


def test_serving_loads_from_registry_aliases(runs, settings) -> None:
    mlflow_settings = settings.model_copy(update={"model_source": ModelSource.MLFLOW})
    models = load_models(mlflow_settings)
    assert models[CHAMPION].version == "1"
    assert models[CHALLENGER].version == "2"


def test_promote_then_rollback(runs, settings) -> None:
    registry = ModelRegistry(settings.registered_model_name)
    registry.promote("2")
    assert registry.aliases() == {CHAMPION: "2", CHALLENGER: None, PREVIOUS: "1", SHADOW: None}
    assert registry.rollback() == "1"
    assert registry.aliases() == {CHAMPION: "1", CHALLENGER: None, PREVIOUS: "2", SHADOW: None}
    exported = registry.export_aliases(settings.artifacts_dir)
    assert exported == {CHAMPION: "1", CHALLENGER: None, SHADOW: None}


def test_rejected_candidate_can_be_shadowed(runs, settings) -> None:
    _, _, third = runs
    assert third.decision.action == "rejected"
    registry = ModelRegistry(settings.registered_model_name)
    registry.set_shadow(third.model_version)
    assert registry.aliases()[SHADOW] == third.model_version
    mlflow_settings = settings.model_copy(update={"model_source": ModelSource.MLFLOW})
    models = load_models(mlflow_settings)
    assert models[SHADOW].version == third.model_version
    assert models[CHAMPION].version == "1"
    registry.set_shadow(None)
    assert registry.aliases()[SHADOW] is None
    assert SHADOW not in load_models(mlflow_settings)
