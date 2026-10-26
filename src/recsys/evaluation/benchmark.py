"""Offline benchmark behind the README results table.

For one dataset: fit every model family on train + val with the hyper-parameters from
``configs/<dataset>.yaml`` (selected on the validation window), evaluate on the untouched
test window, run the streaming-value experiment, and - as a protocol sanity check - repeat
popularity vs. implicit ALS on a *random* 80/20 split to show how much a leaky split
inflates MF gains. Writes ``reports/benchmark_<dataset>.json`` and ``.md``.
"""

from __future__ import annotations

import gc
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from recsys.config import ModelConfig, Settings, load_train_config
from recsys.data.prepare import load_prepared
from recsys.evaluation.evaluate import (
    evaluate_ranking,
    evaluate_rating,
    evaluate_streaming_value,
)
from recsys.evaluation.protocols import leave_one_out_check
from recsys.evaluation.significance import paired_bootstrap
from recsys.models.io import build_model, model_params
from recsys.utils import hardware_info

logger = logging.getLogger(__name__)


def _model_suite(
    cfg_model: ModelConfig,
    baseline: ModelConfig,
    explicit: ModelConfig,
    sequential: ModelConfig | None = None,
    order_ablation: bool = False,
):
    """All models share the tuned popularity baseline as cold-start fallback, so differences
    in the all-users metrics come only from how each model ranks for warm users.

    ``order_ablation`` adds SASRec trained on randomly permuted histories: if it scores as
    well as the time-ordered model, the sequence order is not what the model exploits.
    """
    fallback = {"popularity_half_life_days": baseline.popularity_half_life_days}
    suite = {
        "popularity": baseline,
        "explicit_als": explicit.model_copy(update=fallback),
        "implicit_als": cfg_model.model_copy(update={**fallback, "blend_weight": 0.0}),
    }
    if cfg_model.blend_weight > 0:
        suite["hybrid_als"] = cfg_model.model_copy(update=fallback)
    if sequential is not None:
        suite["sasrec"] = sequential.model_copy(update=fallback)
        if order_ablation:
            suite["sasrec_shuffled"] = sequential.model_copy(
                update={**fallback, "sequence_order": "shuffled"}
            )
    return suite


TRAINING_KEYS = ("best_epoch", "epochs_run", "device", "peak_vram_gb", "onnx_max_abs_diff")


# (A, B): "is A better than B?" - reported when both models are in the run.
COMPARISONS = (
    ("implicit_als", "popularity"),
    ("hybrid_als", "popularity"),
    ("hybrid_als", "implicit_als"),
    ("sasrec", "popularity"),
    ("sasrec", "hybrid_als"),
    ("sasrec_shuffled", "sasrec"),
)


def _fit_eval(
    name: str,
    mc: ModelConfig,
    train: pd.DataFrame,
    test: pd.DataFrame,
    k: int,
    thr: float,
    per_user: dict[int, float] | None = None,
    checkpoint_dir: Path | None = None,
):
    start = time.perf_counter()
    model = build_model(mc)
    if checkpoint_dir is not None and hasattr(model, "set_checkpoint"):
        model.set_checkpoint(checkpoint_dir / f"benchmark-{name}.pt")  # survives interruptions
    model.fit(train)
    fit_seconds = time.perf_counter() - start
    ranking = evaluate_ranking(model, test, k, thr, per_user=per_user)
    rating = evaluate_rating(model, test)
    logger.info(
        "%-13s fit=%.1fs ndcg@%d all=%.4f warm=%.4f cold=%.4f rmse=%s",
        name,
        fit_seconds,
        k,
        ranking["all"]["ndcg"],
        ranking["warm"]["ndcg"],
        ranking["cold"]["ndcg"],
        f"{rating['rmse']:.4f}" if rating else "n/a",
    )
    result = {
        "params": model_params(mc),
        "fit_seconds": round(fit_seconds, 2),
        "ranking": ranking,
        "rating": rating,
    }
    if any(key in model.metadata for key in TRAINING_KEYS):
        result["training"] = {key: model.metadata.get(key) for key in TRAINING_KEYS}
    return model, result


def random_split_check(
    ratings: pd.DataFrame, models: dict[str, ModelConfig], k: int, thr: float, seed: int = 0
) -> dict[str, Any]:
    mask = np.random.default_rng(seed).random(len(ratings)) < 0.8
    train, test = ratings[mask], ratings[~mask]
    out = {}
    for name in ("popularity", "implicit_als"):
        _, result = _fit_eval(f"random/{name}", models[name], train, test, k, thr)
        out[name] = {
            "warm_ndcg": result["ranking"]["warm"]["ndcg"],
            "fit_seconds": result["fit_seconds"],
        }
    out["note"] = "random 80/20 split of all ratings (leaks future interactions into training)"
    return out


def run_benchmark(
    settings: Settings,
    dataset: str | None = None,
    *,
    log_mlflow: bool = True,
    random_check: bool = True,
    models: list[str] | None = None,
    loo_check: bool = False,
    order_ablation: bool = False,
) -> dict[str, Any]:
    """``models`` restricts the run to some models; the report then gets a name suffix so
    the full-suite report is never overwritten by a partial one."""
    dataset = dataset or settings.dataset
    cfg = load_train_config(settings, dataset)
    prepared = load_prepared(settings, dataset)
    train, test = prepared.split.train_val, prepared.split.test
    k, thr = cfg.evaluation.k, cfg.evaluation.relevance_threshold
    full_suite = _model_suite(cfg.model, cfg.baseline, cfg.explicit, cfg.sequential, order_ablation)
    if models:
        unknown = sorted(set(models) - set(full_suite))
        if unknown:
            raise ValueError(f"unknown models {unknown}; configured: {sorted(full_suite)}")
    suite = {name: mc for name, mc in full_suite.items() if not models or name in models}

    results: dict[str, Any] = {}
    fitted = {}
    per_user: dict[str, dict[int, float]] = {}
    checkpoints = settings.artifacts_dir / "checkpoints" / dataset
    for name, mc in suite.items():
        per_user[name] = {}
        fitted[name], results[name] = _fit_eval(
            name, mc, train, test, k, thr, per_user=per_user[name], checkpoint_dir=checkpoints
        )
    # Model differences only exist for returning users (cold users share one fallback), so
    # the comparisons are paired over them; the all-users version is kept for reference.
    returning = set(train["user_id"].unique().tolist())
    pairs = [(a, b) for a, b in COMPARISONS if a in per_user and b in per_user]
    significance = {
        f"{a} vs {b}": paired_bootstrap(per_user[a], per_user[b], users=returning) for a, b in pairs
    }
    significance_all = {f"{a} vs {b}": paired_bootstrap(per_user[a], per_user[b]) for a, b in pairs}

    streaming = {}
    for name in ("popularity", "implicit_als", "hybrid_als", "sasrec", "sasrec_shuffled"):
        if name in fitted:
            streaming[name] = evaluate_streaming_value(
                fitted[name], test, k, thr, cfg.evaluation.context_events
            )

    protocol = {
        "split": "global temporal: train < val < test (fractions from config)",
        "fit_on": "train + val",
        "evaluate_on": "test",
        "k": k,
        "relevance_threshold": thr,
        "context_events": cfg.evaluation.context_events,
        "val_start": prepared.split.val_start,
        "test_start": prepared.split.test_start,
        "n_train_val": len(train),
        "n_test": len(test),
    }
    # The temporal phase is done: release its models and split before the protocol checks,
    # which build their own copies of every rating (25M rows on ml-25m; a 14 GB box ran out
    # of memory holding both).
    all_ratings = pd.concat(
        [prepared.split.train, prepared.split.val, prepared.split.test], ignore_index=True
    )
    fitted.clear()
    del prepared, train, test
    gc.collect()
    random_check_result = (
        random_split_check(all_ratings, full_suite, k, thr) if random_check else None
    )
    gc.collect()
    loo_result = (
        leave_one_out_check(
            all_ratings,
            {name: mc for name, mc in suite.items() if name != "explicit_als"},
            k,
            checkpoint_dir=checkpoints,
        )
        if loo_check
        else None
    )
    report = {
        "dataset": dataset,
        "protocol": protocol,
        "models": results,
        "significance": significance,
        "significance_all_users": significance_all,
        "streaming_value": streaming,
        "random_split_check": random_check_result,
        "leave_one_out_check": loo_result,
        "hardware": hardware_info(),
    }
    out = settings.reports_dir
    out.mkdir(parents=True, exist_ok=True)
    stem = f"benchmark_{dataset}" + (f"_{'+'.join(suite)}" if models else "")
    report["report_name"] = stem
    (out / f"{stem}.json").write_text(json.dumps(report, indent=2, default=str), "utf-8")
    (out / f"{stem}.md").write_text(to_markdown(report), "utf-8")

    if log_mlflow:
        import mlflow

        from recsys.training.tracking import configure_mlflow

        configure_mlflow(settings, experiment="recsys-benchmark")
        for name, res in results.items():
            with mlflow.start_run(run_name=f"{name}-{dataset}"):
                mlflow.set_tags({"dataset": dataset, "model_type": name})
                mlflow.log_params({f"model.{p}": v for p, v in res["params"].items()})
                flat = {
                    f"{seg}_{metric}_at_{k}": value
                    for seg, values in res["ranking"].items()
                    for metric, value in values.items()
                    if metric not in ("n_users",)
                }
                flat |= res["rating"] or {}
                mlflow.log_metrics(
                    {key: float(v) for key, v in flat.items()} | {"fit_seconds": res["fit_seconds"]}
                )
    return report


def _fmt(value: float | None, digits: int = 4) -> str:
    return "n/a" if value is None or value != value else f"{value:.{digits}f}"


def to_markdown(report: dict[str, Any]) -> str:
    k = report["protocol"]["k"]
    lines = [
        f"# Offline benchmark - {report['dataset']}",
        "",
        f"Temporal split, fit on train+val, evaluated on test (K={k}, relevant = rating >= "
        f"{report['protocol']['relevance_threshold']}).",
        "",
        f"| Model | RMSE | P@{k} | R@{k} | NDCG@{k} | MAP@{k} | NDCG@{k} warm "
        f"| NDCG@{k} cold | Coverage | Fit (s) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, res in report["models"].items():
        warm, allu, cold = res["ranking"]["warm"], res["ranking"]["all"], res["ranking"]["cold"]
        rmse = res["rating"]["rmse"] if res["rating"] else None
        lines.append(
            f"| {name} | {_fmt(rmse)} | {_fmt(warm['precision'])} | {_fmt(warm['recall'])} | "
            f"{_fmt(allu['ndcg'])} | {_fmt(warm['map'])} | {_fmt(warm['ndcg'])} | "
            f"{_fmt(cold['ndcg'])} | {_fmt(allu['coverage'], 3)} | {res['fit_seconds']} |"
        )
    any_model = next(iter(report["models"].values()))["ranking"]
    lines += [
        "",
        f"P/R/MAP and 'warm' columns: {any_model['warm']['n_users']} users seen in training; "
        f"'cold': {any_model['cold']['n_users']} users first seen after the cutoff; "
        f"NDCG 'all': {any_model['all']['n_users']} users.",
    ]
    if report.get("significance"):
        lines += [
            "",
            f"## Paired bootstrap over returning test users (NDCG@{k})",
            "",
            "| Comparison | Lift | 95% CI | p (two-sided) | Users |",
            "|---|---|---|---|---|",
        ]
        for name, res in report["significance"].items():
            lo, hi = res["relative_lift_ci95"]
            lines.append(
                f"| {name} | {res['relative_lift']:+.1%} | [{lo:+.1%}, {hi:+.1%}] | "
                f"{res['p_value']:.4f} | {res['n_users']} |"
            )
    lines += [
        "",
        "## Streaming value (first "
        f"{report['protocol']['context_events']} test events of each user used as "
        "real-time context)",
        "",
        f"| Model | Segment | NDCG@{k} batch-only | NDCG@{k} streaming | Users |",
        "|---|---|---|---|---|",
    ]
    for name, segments in report["streaming_value"].items():
        for segment in ("warm", "cold", "all"):
            batch = segments[segment]["batch_only"]
            stream = segments[segment]["streaming"]
            lines.append(
                f"| {name} | {segment} | {_fmt(batch['ndcg'])} | {_fmt(stream['ndcg'])} | "
                f"{batch['n_users']} |"
            )
    if report.get("random_split_check"):
        rc = report["random_split_check"]
        lines += [
            "",
            "## Protocol check: random 80/20 split (leaky)",
            "",
            f"| Model | NDCG@{k} (warm) |",
            "|---|---|",
            f"| popularity | {_fmt(rc['popularity']['warm_ndcg'])} |",
            f"| implicit_als | {_fmt(rc['implicit_als']['warm_ndcg'])} |",
        ]
    if report.get("leave_one_out_check"):
        loo = report["leave_one_out_check"]
        lines += [
            "",
            "## Protocol check: leave-one-out (common in papers; leaks the global timeline)",
            "",
            f"| Model | NDCG@{k} | HR@{k} | Users |",
            "|---|---|---|---|",
        ]
        for name, res in loo.items():
            if name != "note":
                lines.append(
                    f"| {name} | {_fmt(res['ndcg'])} | {_fmt(res['hit_rate'])} | {res['n_users']} |"
                )
    hw = report["hardware"]
    lines += [
        "",
        f"Hardware: {hw['cpu']}, {hw['logical_cpus']} logical CPUs, {hw['ram_gb']} GB RAM, "
        f"{hw['os']}.",
    ]
    return "\n".join(lines) + "\n"
