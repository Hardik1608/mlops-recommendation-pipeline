"""``recsys`` command-line interface - one entry point for local runs, Docker, Airflow and CI.

Typical local flow::

    recsys download                # MovieLens ml-latest-small (checksum verified)
    recsys prepare                 # validate + temporal split
    recsys features                # batch feature tables for Feast
    recsys train                   # fit, evaluate vs baseline, register, gate
    recsys serve                   # FastAPI on :8000
    recsys stream                  # replay test-window events (in-memory broker) -> features
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from recsys.config import Settings, load_train_config
from recsys.logging_utils import configure_logging

logger = logging.getLogger("recsys.cli")


def _parse_grid(items: list[str]) -> dict[str, list[Any]]:
    """``--grid alpha=1,5,20 factors=32,64`` -> ``{"alpha": [1, 5, 20], "factors": [32, 64]}``."""

    def convert(value: str) -> Any:
        if value.lower() in {"none", "null"}:
            return None
        for cast in (int, float):
            try:
                return cast(value)
            except ValueError:
                continue
        return value

    grid: dict[str, list[Any]] = {}
    for item in items:
        key, _, values = item.partition("=")
        grid[key] = [convert(v) for v in values.split(",")]
    return grid


def _dump(obj: Any) -> None:
    print(json.dumps(obj, indent=2, default=str))


# ------------------------------------------------------------------------------ data
def cmd_download(args: argparse.Namespace, settings: Settings) -> None:
    from recsys.data.download import download_dataset

    print(download_dataset(args.dataset, settings.raw_dir, force=args.force))


def cmd_synth(args: argparse.Namespace, settings: Settings) -> None:
    from recsys.data.synthetic import write_synthetic

    out = write_synthetic(
        settings.raw_dir / "synthetic", n_users=args.users, n_items=args.items, seed=args.seed
    )
    print(out)


def cmd_prepare(args: argparse.Namespace, settings: Settings) -> None:
    from recsys.data.prepare import ensure_raw, prepare_dataset

    ensure_raw(settings, args.dataset)
    cfg = load_train_config(settings, args.dataset)
    _dump(prepare_dataset(settings, args.dataset, cfg.split).info)


def cmd_features(args: argparse.Namespace, settings: Settings) -> None:
    from recsys.data.prepare import load_prepared
    from recsys.features.offline import write_feature_tables

    prepared = load_prepared(settings, args.dataset)
    paths = write_feature_tables(prepared.split.train_val, prepared.movies, settings.features_dir)
    _dump({name: str(path) for name, path in paths.items()})


# ------------------------------------------------------------------------------ feast
def cmd_feast_apply(args: argparse.Namespace, settings: Settings) -> None:
    from recsys.pipelines.tasks import feast_apply

    _dump(feast_apply(settings))


def cmd_feast_materialize(args: argparse.Namespace, settings: Settings) -> None:
    from recsys.pipelines.tasks import feast_materialize

    _dump(feast_materialize(settings))


# ------------------------------------------------------------------------------ modelling
def cmd_tune(args: argparse.Namespace, settings: Settings) -> None:
    from recsys.training.tune import run_grid

    summary = run_grid(
        settings,
        args.dataset,
        args.model,
        _parse_grid(args.grid),
        metric=args.metric,
        log_mlflow=args.mlflow,
        name=args.name,
    )
    _dump(summary["best"])


def cmd_tune_plan(args: argparse.Namespace, settings: Settings) -> None:
    from recsys.training.tune import run_plan

    _dump(
        run_plan(settings, args.plan, log_mlflow=args.mlflow, start=args.start, resume=args.resume)
    )


def cmd_train(args: argparse.Namespace, settings: Settings) -> None:
    from recsys.config import TrainConfig
    from recsys.training.train import summarize, train_and_register

    config = TrainConfig.from_yaml(args.config) if args.config else None
    result = train_and_register(
        settings,
        config.dataset if config else args.dataset,
        register=not args.no_register,
        export_local=not args.no_export,
        config=config,
    )
    _dump(summarize(result))


def cmd_benchmark(args: argparse.Namespace, settings: Settings) -> None:
    from recsys.evaluation.benchmark import run_benchmark

    report = run_benchmark(
        settings,
        args.dataset,
        log_mlflow=not args.no_mlflow,
        random_check=not args.no_random,
        models=args.models.split(",") if args.models else None,
        loo_check=args.loo,
        order_ablation=args.order_ablation,
    )
    print((settings.reports_dir / f"{report['report_name']}.md").read_text("utf-8"))


def _registry(settings: Settings):
    import mlflow

    from recsys.training.registry import ModelRegistry

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    return ModelRegistry(settings.registered_model_name)


def cmd_export_demo(args: argparse.Namespace, settings: Settings) -> None:
    from recsys.evaluation.demo_export import export_demo

    payload = export_demo(
        settings,
        args.dataset or settings.dataset,
        Path(args.out),
        n_users=args.users,
        include_sequential=not args.no_sequential,
    )
    _dump({"users": len(payload["users"]), "movies": len(payload["movies"]), "out": args.out})


def cmd_promote(args: argparse.Namespace, settings: Settings) -> None:
    from recsys.training.registry import CHALLENGER

    registry = _registry(settings)
    version = args.version or registry.version_for(CHALLENGER)
    if version is None:
        raise SystemExit("no --version given and no challenger alias set")
    registry.promote(str(version))
    _dump(
        {"aliases": registry.aliases(), "exported": registry.export_aliases(settings.artifacts_dir)}
    )


def cmd_rollback(args: argparse.Namespace, settings: Settings) -> None:
    registry = _registry(settings)
    registry.rollback()
    _dump(
        {"aliases": registry.aliases(), "exported": registry.export_aliases(settings.artifacts_dir)}
    )


def cmd_aliases(args: argparse.Namespace, settings: Settings) -> None:
    _dump(_registry(settings).aliases())


def cmd_shadow(args: argparse.Namespace, settings: Settings) -> None:
    if (args.version is None) == (not args.clear):
        raise SystemExit("give either --version N or --clear")
    registry = _registry(settings)
    registry.set_shadow(None if args.clear else str(args.version))
    _dump(
        {"aliases": registry.aliases(), "exported": registry.export_aliases(settings.artifacts_dir)}
    )


# ------------------------------------------------------------------------------ streaming
def _replay_frame(settings: Settings, dataset: str, window: str):
    from recsys.data.prepare import load_prepared

    prepared = load_prepared(settings, dataset)
    return {"test": prepared.split.test, "val": prepared.split.val}[window]


def cmd_produce(args: argparse.Namespace, settings: Settings) -> None:
    from recsys.streaming.kafka_io import KafkaEventPublisher
    from recsys.streaming.producer import iter_events, replay

    publisher = KafkaEventPublisher(settings.kafka_bootstrap_servers)
    frame = _replay_frame(settings, args.dataset, args.window)
    sent = replay(
        publisher, settings.kafka_topic, iter_events(frame), rate=args.rate, limit=args.limit
    )
    _dump({"published": sent, "topic": settings.kafka_topic})


def _processor(settings: Settings, invalidate: bool):
    from recsys.features.realtime import UserActivityAggregator
    from recsys.features.store import create_feature_store
    from recsys.streaming.processor import StreamProcessor
    from recsys.streaming.sinks import ParquetEventSink

    on_update = None
    if invalidate and settings.cache_backend == "redis":
        from recsys.serving.cache import RedisInvalidator

        on_update = RedisInvalidator(settings.redis_url)
    return StreamProcessor(
        UserActivityAggregator(create_feature_store(settings), settings.recent_items_max),
        sink=ParquetEventSink(
            settings.events_dir, settings.sink_flush_events, settings.sink_flush_seconds
        ),
        on_users_updated=on_update,
    )


def cmd_consume(args: argparse.Namespace, settings: Settings) -> None:
    from recsys.streaming.kafka_io import KafkaEventPublisher, KafkaEventSource
    from recsys.streaming.processor import run_consumer

    dlq = KafkaEventPublisher(settings.kafka_bootstrap_servers, client_id="recsys-dlq")
    processor = _processor(settings, invalidate=True)
    processor.dead_letter = lambda raw, err: dlq.publish(
        settings.kafka_dlq_topic, b"invalid", raw if isinstance(raw, bytes) else str(raw).encode()
    )
    source = KafkaEventSource(
        settings.kafka_bootstrap_servers, settings.kafka_topic, settings.kafka_group_id
    )
    total = run_consumer(source, processor, max_idle_polls=args.max_idle_polls)
    dlq.flush()
    _dump(vars(total))


def cmd_stream(args: argparse.Namespace, settings: Settings) -> None:
    """Broker-less replay: producer -> in-memory log -> processor -> feature store + sink."""
    from recsys.streaming.bus import InMemoryBroker
    from recsys.streaming.processor import run_consumer
    from recsys.streaming.producer import iter_events, replay

    broker = InMemoryBroker()
    frame = _replay_frame(settings, args.dataset, args.window)
    sent = replay(
        broker, settings.kafka_topic, iter_events(frame), rate=args.rate, limit=args.limit
    )
    processor = _processor(settings, invalidate=True)
    total = run_consumer(
        broker.consumer(settings.kafka_topic, settings.kafka_group_id),
        processor,
        poll_timeout=0.0,
        max_idle_polls=1,
    )
    _dump({"published": sent, **vars(total)})


# ------------------------------------------------------------------------------ ops
def cmd_drift(args: argparse.Namespace, settings: Settings) -> None:
    from recsys.monitoring.drift import run_drift_job

    _dump(run_drift_job(settings, args.dataset, source=args.source))


def cmd_serve(args: argparse.Namespace, settings: Settings) -> None:
    import uvicorn

    uvicorn.run(
        "recsys.serving.app:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        log_level=settings.log_level.lower(),
        access_log=False,
    )


def cmd_loadtest(args: argparse.Namespace, settings: Settings) -> None:
    from recsys.loadtest import main as loadtest_main

    loadtest_main(args.loadtest_args)


# ------------------------------------------------------------------------------ parser
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="recsys", description="MLOps recommendation pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name: str, func: Any, help_text: str) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=help_text)
        p.set_defaults(func=func)
        return p

    p = add("download", cmd_download, "download + verify a MovieLens archive")
    p.add_argument("--dataset", default=None)
    p.add_argument("--force", action="store_true")

    p = add("synth", cmd_synth, "generate the synthetic dataset (tests / CI)")
    p.add_argument("--users", type=int, default=300)
    p.add_argument("--items", type=int, default=500)
    p.add_argument("--seed", type=int, default=7)

    p = add("prepare", cmd_prepare, "validate + temporally split a dataset")
    p.add_argument("--dataset", default=None)

    p = add("features", cmd_features, "write batch feature tables (Feast sources)")
    p.add_argument("--dataset", default=None)

    add("feast-apply", cmd_feast_apply, "register feature definitions (feast apply)")
    add("feast-materialize", cmd_feast_materialize, "load batch features into the online store")

    p = add("tune", cmd_tune, "grid search on the validation window")
    p.add_argument("--dataset", default=None)
    p.add_argument("--model", default="implicit_als")
    p.add_argument("--metric", default="warm_ndcg_at_10")
    p.add_argument("--grid", nargs="+", required=True, help="param=v1,v2 ...")
    p.add_argument("--mlflow", action="store_true", help="log every trial to MLflow")
    p.add_argument("--name", default=None, help="report name (default: model type)")

    p = add("export-demo", cmd_export_demo, "precomputed recommendations for the static demo")
    p.add_argument("--dataset", default=None)
    p.add_argument("--out", default="reports/demo/demo.json")
    p.add_argument("--users", type=int, default=60)
    p.add_argument("--no-sequential", action="store_true", help="skip SASRec (no torch/GPU)")

    p = add("tune-plan", cmd_tune_plan, "staged grid search from a YAML plan")
    p.add_argument("--plan", required=True, help="e.g. configs/tuning/ml-25m.yaml")
    p.add_argument("--mlflow", action="store_true", help="log every trial to MLflow")
    p.add_argument("--start", type=int, default=1, help="resume from this stage number")
    p.add_argument(
        "--resume",
        action="store_true",
        help="skip stages with a report; continue an interrupted stage where it stopped",
    )

    p = add("train", cmd_train, "train, evaluate vs baseline, register + gate in MLflow")
    p.add_argument("--dataset", default=None)
    p.add_argument("--config", default=None, help="alternative training YAML (experiments)")
    p.add_argument("--no-register", action="store_true")
    p.add_argument("--no-export", action="store_true")

    p = add("benchmark", cmd_benchmark, "offline benchmark of all model families")
    p.add_argument("--dataset", default=None)
    p.add_argument("--no-mlflow", action="store_true")
    p.add_argument("--no-random", action="store_true", help="skip the random-split check")
    p.add_argument("--models", default=None, help="comma-separated subset, e.g. sasrec")
    p.add_argument("--loo", action="store_true", help="add the leave-one-out protocol check")
    p.add_argument(
        "--order-ablation", action="store_true", help="add SASRec trained on shuffled histories"
    )

    p = add("promote", cmd_promote, "promote a version (default: challenger) to champion")
    p.add_argument("--version", default=None)
    add("rollback", cmd_rollback, "restore previous_champion")
    add("aliases", cmd_aliases, "show registry aliases")

    p = add("shadow", cmd_shadow, "set or clear the shadow alias (scored, never served)")
    p.add_argument("--version", default=None)
    p.add_argument("--clear", action="store_true")

    for name, func, text in (
        ("produce", cmd_produce, "replay a split window to Kafka"),
        ("stream", cmd_stream, "broker-less replay into the feature store (local demo)"),
    ):
        p = add(name, func, text)
        p.add_argument("--dataset", default=None)
        p.add_argument("--window", choices=["test", "val"], default="test")
        p.add_argument("--rate", type=float, default=None, help="events per second")
        p.add_argument("--limit", type=int, default=None)

    p = add("consume", cmd_consume, "Kafka -> online features + offline event store")
    p.add_argument("--max-idle-polls", type=int, default=None)

    p = add("drift", cmd_drift, "PSI drift report (reference = training window)")
    p.add_argument("--dataset", default=None)
    p.add_argument("--source", choices=["events", "test"], default="events")

    p = add("serve", cmd_serve, "run the API with uvicorn")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)

    p = add("loadtest", cmd_loadtest, "HTTP load test (see recsys.loadtest --help)")
    p.add_argument("loadtest_args", nargs=argparse.REMAINDER)
    return parser


def main(argv: list[str] | None = None) -> int:
    settings = Settings()
    configure_logging(settings.log_level, settings.log_format)
    args = build_parser().parse_args(argv)
    if getattr(args, "dataset", "unset") is None:
        args.dataset = settings.dataset
    args.func(args, settings)
    return 0


if __name__ == "__main__":
    sys.exit(main())
