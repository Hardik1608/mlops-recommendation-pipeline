"""Grid search on the *validation* window (train -> val). The test window is never touched.

Results are written to ``reports/tuning/<dataset>/<name>.json`` and (optionally) logged to
MLflow as child runs, so the chosen hyper-parameters in ``configs/<dataset>.yaml`` are
traceable to the run that selected them.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import logging
import time
from pathlib import Path
from typing import Any

from recsys.config import ModelConfig, Settings, load_train_config
from recsys.data.prepare import load_prepared
from recsys.evaluation.evaluate import evaluate_ranking, evaluate_rating, flatten_metrics
from recsys.models.io import build_model

logger = logging.getLogger(__name__)

LOWER_IS_BETTER = {"rmse", "mae"}


def _base_config(model_type: str, settings: Settings, dataset: str) -> ModelConfig:
    cfg = load_train_config(settings, dataset)
    for candidate in (cfg.model, cfg.baseline, cfg.explicit, cfg.sequential):
        if candidate is not None and candidate.type == model_type:
            return candidate
    return ModelConfig(type=model_type)  # type: ignore[arg-type]


def run_grid(
    settings: Settings,
    dataset: str,
    model_type: str,
    grid: dict[str, list[Any]],
    metric: str,
    log_mlflow: bool = False,
    name: str | None = None,
    reuse: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Grid search on the validation window.

    ``reuse`` maps full model configurations to results from earlier stages of a plan: a
    trial whose configuration already ran (e.g. the previous stage's winner re-entering a
    later grid) is reused instead of trained again. Finished trials are added to it.
    """
    prepared = load_prepared(settings, dataset)
    cfg = load_train_config(settings, dataset)
    k, threshold = cfg.evaluation.k, cfg.evaluation.relevance_threshold
    base = _base_config(model_type, settings, dataset)
    keys = list(grid)
    results: list[dict[str, Any]] = []
    out_dir = settings.reports_dir / "tuning" / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    report_name = name or model_type
    # Finished trials survive an interruption: each one is appended to a partial report
    # and reused on the next run, but only if nothing that defines a trial has changed.
    partial_path = out_dir / f"{report_name}.partial.json"
    fingerprint = {
        "dataset": dataset,
        "model_type": model_type,
        "metric": metric,
        "k": k,
        "threshold": threshold,
        "base": base.model_dump(mode="json"),
        "split": [prepared.split.val_start, prepared.split.test_start],
    }
    finished: dict[str, dict[str, Any]] = {}
    if partial_path.exists():
        partial = json.loads(partial_path.read_text(encoding="utf-8"))
        if partial.get("fingerprint") == fingerprint:
            finished = {_trial_key(r["params"]): r for r in partial["results"]}
            logger.info("resuming %s: %d finished trials", report_name, len(finished))
        else:
            logger.warning("ignoring %s: it was written for a different setup", partial_path)

    parent = None
    if log_mlflow:
        import mlflow

        from recsys.training.tracking import configure_mlflow

        configure_mlflow(settings, experiment="recsys-tuning")
        parent = mlflow.start_run(run_name=f"tune-{model_type}-{dataset}")
        mlflow.log_params({"dataset": dataset, "model_type": model_type, "metric": metric})

    try:
        for combo in itertools.product(*(grid[key] for key in keys)):
            params = dict(zip(keys, combo, strict=True))
            trial = _trial_key(params)
            if trial in finished:
                results.append(finished[trial])
                logger.info("%s %s -> reused from %s", model_type, params, partial_path.name)
                continue
            model_cfg = base.model_copy(update={**params, "type": model_type})
            config_key = _trial_key(model_cfg.model_dump(mode="json"))
            if reuse is not None and config_key in reuse:
                results.append({**reuse[config_key], "params": params})
                _write_json(partial_path, {"fingerprint": fingerprint, "results": results})
                logger.info("%s %s -> same configuration as an earlier stage", model_type, params)
                continue
            model = build_model(model_cfg)
            if hasattr(model, "set_checkpoint"):  # resumable epochs (GPU runs take hours)
                digest = hashlib.sha1(trial.encode()).hexdigest()[:12]
                checkpoints = settings.artifacts_dir / "checkpoints" / dataset
                model.set_checkpoint(checkpoints / f"{report_name}-{digest}.pt")
            if hasattr(model, "set_monitor"):
                # Early stopping on the same validation window; the best epoch is recorded
                # and becomes ``epochs`` in the training config (the test window is unseen).
                model.set_monitor(
                    lambda m: flatten_metrics(
                        evaluate_ranking(m, prepared.split.val, k, threshold), None, k
                    )[metric]
                )
            start = time.perf_counter()
            model.fit(prepared.split.train)
            fit_seconds = time.perf_counter() - start
            flat = flatten_metrics(
                evaluate_ranking(model, prepared.split.val, k, threshold),
                evaluate_rating(model, prepared.split.val),
                k,
            )
            result: dict[str, Any] = {"params": params, "metrics": flat, "fit_seconds": fit_seconds}
            history = getattr(model, "history", [])
            if model.model_type == "sasrec" and history:
                result["training"] = {
                    key: model.metadata.get(key)
                    for key in (
                        "best_epoch",
                        "epochs_run",
                        "device",
                        "peak_vram_gb",
                        "onnx_max_abs_diff",
                        "train_sequences",
                    )
                } | {
                    "seconds_per_epoch": sum(h["seconds"] for h in history) / len(history),
                    "curve": [
                        {key: h[key] for key in ("iteration", "loss", "val_metric") if key in h}
                        for h in history
                    ],
                }
            results.append(result)
            if reuse is not None:
                reuse[config_key] = result
            _write_json(partial_path, {"fingerprint": fingerprint, "results": results})
            logger.info(
                "%s %s -> %s=%.4f (%.1fs)", model_type, params, metric, flat[metric], fit_seconds
            )
            if log_mlflow:
                import mlflow

                with mlflow.start_run(run_name=str(params), nested=True):
                    mlflow.log_params({**params, "model_type": model_type})
                    mlflow.log_metrics({**flat, "fit_seconds": fit_seconds})
    finally:
        if parent is not None:
            import mlflow

            mlflow.end_run()

    reverse = metric not in LOWER_IS_BETTER
    ranked = sorted(results, key=lambda r: r["metrics"][metric], reverse=reverse)
    summary = {
        "dataset": dataset,
        "model_type": model_type,
        "selection_metric": metric,
        "split": "train -> val",
        "grid": grid,
        "best": ranked[0],
        "results": ranked,
    }
    path = out_dir / f"{report_name}.json"
    _write_json(path, summary)
    partial_path.unlink(missing_ok=True)
    logger.info("Best %s on %s: %s -> %s", model_type, dataset, ranked[0]["params"], path)
    return summary


def _trial_key(params: dict[str, Any]) -> str:
    return json.dumps(params, sort_keys=True, default=str)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write via a temporary file and rename, so an interruption never truncates it."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def run_plan(
    settings: Settings,
    plan_path: str | Path,
    log_mlflow: bool = False,
    start: int = 1,
    resume: bool = False,
) -> dict[str, Any]:
    """Run a staged search: each stage is a grid; later stages of the same model keep the
    best values of earlier ones fixed (``configs/tuning/<plan>.yaml``).

    Stages before ``start`` are not re-run: their saved reports are read back. With
    ``resume``, every stage whose report exists is read back too, and an interrupted stage
    continues from its finished trials (and SASRec from its last epoch checkpoint).
    SASRec's early-stopping epoch is carried forward as ``epochs``.
    The selected values per model go to ``reports/tuning/<dataset>/selected_<plan>.json``.
    """
    import yaml

    plan = yaml.safe_load(Path(plan_path).read_text(encoding="utf-8"))
    dataset = plan["dataset"]
    out_dir = settings.reports_dir / "tuning" / dataset
    fixed: dict[str, dict[str, Any]] = {}  # carried into later grids of the same model
    reuse: dict[str, dict[str, Any]] = {}  # full config -> result, shared across stages
    selected: dict[str, dict[str, Any]] = {}  # what the training config should use
    for number, stage in enumerate(plan["stages"], start=1):
        model, name = stage["model"], stage["name"]
        carried = fixed.get(model, {})
        report = out_dir / f"{name}.json"
        if number < start or (resume and report.exists()):
            summary = json.loads(report.read_text(encoding="utf-8"))
            logger.info(
                "stage %d/%d %s: read back from %s", number, len(plan["stages"]), name, report.name
            )
            base = _base_config(model, settings, dataset)
            for result in summary["results"]:
                trial_cfg = base.model_copy(update={**result["params"], "type": model})
                reuse[_trial_key(trial_cfg.model_dump(mode="json"))] = result
        else:
            grid = {key: [value] for key, value in carried.items() if key not in stage["grid"]}
            grid |= stage["grid"]
            logger.info("stage %d/%d %s: %s", number, len(plan["stages"]), name, grid)
            summary = run_grid(
                settings, dataset, model, grid, stage["metric"], log_mlflow, name=name, reuse=reuse
            )
        best = summary["best"]
        fixed[model] = {**carried, **best["params"]}
        selected[model] = {**fixed[model], "selected_by": f"{name} ({stage['metric']})"}
        # Early stopping picks the epoch count per trial; it is not fixed for later stages
        # (that would cap their search), only reported for the final training run.
        best_epoch = (best.get("training") or {}).get("best_epoch")
        if best_epoch:
            selected[model]["epochs"] = int(best_epoch)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"selected_{Path(plan_path).stem}.json"
    path.write_text(json.dumps(selected, indent=2), encoding="utf-8")
    return selected
