"""Training pipeline: fit -> evaluate vs baseline -> log to MLflow -> register -> gate.

Protocol (no leakage): hyper-parameters were chosen on train -> val (``recsys tune``);
the candidate is fit on train + val and evaluated on the untouched test window. The
current champion is re-evaluated on the *same* test window, so the gate compares like
with like even when the champion was trained on older data.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from recsys.config import Settings, TrainConfig, load_train_config
from recsys.data.prepare import load_prepared
from recsys.evaluation.evaluate import evaluate_ranking, evaluate_rating, flatten_metrics
from recsys.models.base import Recommender
from recsys.models.io import build_model, model_params
from recsys.models.package import load_package, save_package
from recsys.training.registry import CHAMPION, GateDecision, ModelRegistry, decide

logger = logging.getLogger(__name__)


@dataclass
class TrainResult:
    run_id: str
    model_version: str | None
    decision: GateDecision | None
    metrics: dict[str, float]
    baseline_metrics: dict[str, float]
    package_dir: Path
    exported: dict[str, str | None]


def evaluate(model: Recommender, test: pd.DataFrame, cfg: TrainConfig) -> dict[str, float]:
    k, threshold = cfg.evaluation.k, cfg.evaluation.relevance_threshold
    return flatten_metrics(
        evaluate_ranking(model, test, k, threshold), evaluate_rating(model, test), k
    )


def _pip_requirements(model_type: str) -> list[str]:
    from importlib.metadata import version

    pkgs = ["numpy", "scipy", "pandas", "pyarrow", "mlflow"]
    if model_type == "sasrec":
        pkgs.append("onnxruntime")  # SASRec is scored from its ONNX export
    return [f"{name}=={version(name)}" for name in pkgs]


def train_and_register(
    settings: Settings,
    dataset: str | None = None,
    *,
    register: bool = True,
    export_local: bool = True,
    config: TrainConfig | None = None,
) -> TrainResult:
    import mlflow
    import mlflow.pyfunc
    from mlflow.models import infer_signature

    import recsys
    from recsys.training.pyfunc import RecsysPyFunc, recommend_frame
    from recsys.training.tracking import configure_mlflow

    dataset = dataset or settings.dataset
    cfg = config or load_train_config(settings, dataset)
    prepared = load_prepared(settings, dataset)
    train_df, test_df = prepared.split.train_val, prepared.split.test
    gate_metric = cfg.gate.metric

    configure_mlflow(settings)
    with mlflow.start_run(run_name=f"{cfg.model.type}-{dataset}") as run:
        run_id = run.info.run_id
        mlflow.set_tags({"dataset": dataset, "model_type": cfg.model.type, "pipeline": "train"})
        mlflow.log_params(
            {f"model.{key}": value for key, value in model_params(cfg.model).items()}
            | {
                "dataset": dataset,
                "ratings_sha256": prepared.info.get("ratings_sha256"),
                "train_rows": len(train_df),
                "test_rows": len(test_df),
                "test_start": prepared.split.test_start,
                "eval.k": cfg.evaluation.k,
                "eval.relevance_threshold": cfg.evaluation.relevance_threshold,
                "baseline.popularity_half_life_days": cfg.baseline.popularity_half_life_days,
            }
        )

        baseline = build_model(cfg.baseline).fit(train_df)
        baseline_metrics = evaluate(baseline, test_df, cfg)
        mlflow.log_metrics({f"baseline_{k}": v for k, v in baseline_metrics.items()})

        model = build_model(cfg.model).fit(train_df)
        metrics = evaluate(model, test_df, cfg)
        mlflow.log_metrics(metrics | {"fit_seconds": float(model.metadata["fit_seconds"])})
        for step in getattr(model, "history", []):
            it = int(step["iteration"])
            mlflow.log_metric("iteration_seconds", step["seconds"], step=it)
            if "loss" in step:
                mlflow.log_metric("train_loss", step["loss"], step=it)
        extra = {
            key: float(model.metadata[key])
            for key in ("peak_vram_gb", "onnx_max_abs_diff", "onnx_mb")
            if model.metadata.get(key) is not None
        }
        if extra:
            mlflow.log_metrics(extra)
        if baseline_metrics.get(gate_metric):
            lift = metrics[gate_metric] / baseline_metrics[gate_metric] - 1.0
            mlflow.log_metric(f"lift_vs_baseline_{gate_metric}", lift)

        package_dir = settings.artifacts_dir / "runs" / run_id
        save_package(
            model,
            prepared.movies,
            package_dir,
            info={
                "run_id": run_id,
                "dataset": dataset,
                "metrics": metrics,
                "baseline_metrics": baseline_metrics,
                "split": {
                    "val_start": prepared.split.val_start,
                    "test_start": prepared.split.test_start,
                },
            },
        )
        logger.info(
            "%s: %s=%.4f (baseline %.4f)",
            cfg.model.type,
            gate_metric,
            metrics[gate_metric],
            baseline_metrics.get(gate_metric, float("nan")),
        )

        version: str | None = None
        decision: GateDecision | None = None
        exported: dict[str, str | None] = {}
        if register:
            # user ids are integers by design; MLflow's "integer column" hint does not apply
            warnings.filterwarnings("ignore", message="Hint: Inferred schema contains integer")
            example = pd.DataFrame({"user_id": model.user_index.ids[:3].astype("int64")})
            signature = infer_signature(
                example,
                recommend_frame(model, example["user_id"], cfg.evaluation.k),
                params={"k": cfg.evaluation.k},
            )
            info = mlflow.pyfunc.log_model(
                name="model",
                python_model=RecsysPyFunc(),
                artifacts={"package": str(package_dir)},
                code_paths=[str(Path(recsys.__file__).parent)],
                pip_requirements=_pip_requirements(cfg.model.type),
                signature=signature,
                input_example=example,
                registered_model_name=settings.registered_model_name,
            )
            version = str(info.registered_model_version)
            registry = ModelRegistry(settings.registered_model_name)
            decision = gate(registry, version, metrics, test_df, cfg, settings)
            registry.apply_decision(version, decision)
            mlflow.set_tags({"gate_action": decision.action, "registered_version": version})
            mlflow.log_dict(decision.to_dict(), "gate_decision.json")
            if export_local:
                exported = registry.export_aliases(settings.artifacts_dir)

    return TrainResult(
        run_id=run_id,
        model_version=version,
        decision=decision,
        metrics=metrics,
        baseline_metrics=baseline_metrics,
        package_dir=package_dir,
        exported=exported,
    )


def gate(
    registry: ModelRegistry,
    version: str,
    metrics: dict[str, float],
    test_df: pd.DataFrame,
    cfg: TrainConfig,
    settings: Settings,
) -> GateDecision:
    """Compare the candidate with the *current* champion on the same test window."""
    champion_version = registry.version_for(CHAMPION)
    champion_value: float | None = None
    if champion_version is not None and champion_version != version:
        champion_dir = settings.artifacts_dir / "gate" / f"v{champion_version}"
        registry.download_package(champion_version, champion_dir)
        champion = load_package(champion_dir).model
        champion_value = evaluate(champion, test_df, cfg)[cfg.gate.metric]
    return decide(
        cfg.gate.metric,
        metrics[cfg.gate.metric],
        champion_value,
        cfg.gate.min_relative_improvement,
        cfg.gate.bootstrap_champion,
    )


def summarize(result: TrainResult) -> dict[str, Any]:
    return {
        "run_id": result.run_id,
        "model_version": result.model_version,
        "decision": result.decision.to_dict() if result.decision else None,
        "metrics": result.metrics,
        "baseline_metrics": result.baseline_metrics,
        "exported": result.exported,
    }
