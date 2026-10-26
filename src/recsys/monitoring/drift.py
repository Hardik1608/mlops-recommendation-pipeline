"""Population Stability Index (PSI) drift report: training reference vs a recent window.

PSI = sum((a_i - e_i) * ln(a_i / e_i)) over bins; rule of thumb: < 0.1 stable,
0.1 - 0.2 moderate shift, > 0.2 significant shift (alert + consider retraining).

Monitored distributions (all derived from interaction events, so they work for any window):

* ``rating``          - rating values (label / feedback distribution)
* ``item_popularity`` - decile of each consumed item's *training* popularity, plus a
                        ``new_item`` bucket for items the model has never seen (catalog drift)
* ``genre``           - primary genre of consumed items (taste drift)
* ``hour_of_day``     - UTC hour of the event (traffic-pattern drift)
"""

from __future__ import annotations

import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from recsys.config import Settings
from recsys.data.prepare import load_prepared
from recsys.streaming.sinks import read_events

logger = logging.getLogger(__name__)


def psi_from_counts(expected: np.ndarray, actual: np.ndarray, eps: float = 1e-4) -> float:
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    if expected.sum() == 0 or actual.sum() == 0:
        return float("nan")
    e = np.clip(expected / expected.sum(), eps, None)
    a = np.clip(actual / actual.sum(), eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_numeric(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    edges = np.unique(np.quantile(np.asarray(reference, dtype=float), np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    return psi_from_counts(np.histogram(reference, edges)[0], np.histogram(current, edges)[0])


def psi_categorical(reference: pd.Series, current: pd.Series) -> float:
    categories = sorted(set(reference.astype(str)) | set(current.astype(str)))
    e = reference.astype(str).value_counts().reindex(categories, fill_value=0)
    a = current.astype(str).value_counts().reindex(categories, fill_value=0)
    return psi_from_counts(e.to_numpy(), a.to_numpy())


def _popularity_bucket(reference: pd.DataFrame, events: pd.DataFrame) -> pd.Series:
    counts = reference["item_id"].value_counts()
    deciles = pd.qcut(counts.rank(method="first"), 10, labels=[f"d{i}" for i in range(10)])
    return events["item_id"].map(deciles).astype(object).fillna("new_item")


def _primary_genre(events: pd.DataFrame, movies: pd.DataFrame) -> pd.Series:
    genres = movies.set_index("item_id")["genres"].str.split("|").str[0]
    return events["item_id"].map(genres).fillna("unknown")


def _hour(events: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(events["timestamp"], unit="s", utc=True).dt.hour


def drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    movies: pd.DataFrame,
    threshold: float = 0.2,
) -> dict[str, Any]:
    features = {
        "rating": psi_categorical(reference["rating"], current["rating"]),
        "item_popularity": psi_categorical(
            _popularity_bucket(reference, reference), _popularity_bucket(reference, current)
        ),
        "genre": psi_categorical(
            _primary_genre(reference, movies), _primary_genre(current, movies)
        ),
        "hour_of_day": psi_categorical(_hour(reference), _hour(current)),
    }
    new_item_share = float((_popularity_bucket(reference, current) == "new_item").mean())
    return {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "threshold": threshold,
        "n_reference": len(reference),
        "n_current": len(current),
        "current_window": [int(current["timestamp"].min()), int(current["timestamp"].max())],
        "new_item_share": new_item_share,
        "features": {
            name: {"psi": round(value, 5), "drift": bool(value > threshold)}
            for name, value in features.items()
        },
        "drift_detected": any(value > threshold for value in features.values()),
    }


def push_drift_metrics(url: str, report: dict[str, Any], job: str = "recsys_drift") -> None:
    from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

    registry = CollectorRegistry()
    psi = Gauge(
        "recsys_drift_psi", "PSI per monitored distribution", ["feature"], registry=registry
    )
    for name, values in report["features"].items():
        psi.labels(name).set(values["psi"])
    Gauge("recsys_drift_detected", "1 if any PSI exceeds the threshold", registry=registry).set(
        int(report["drift_detected"])
    )
    Gauge("recsys_drift_new_item_share", "Share of events on unseen items", registry=registry).set(
        report["new_item_share"]
    )
    Gauge("recsys_drift_last_run_timestamp_seconds", "Last drift run", registry=registry).set(
        time.time()
    )
    push_to_gateway(url, job=job, registry=registry)


def run_drift_job(
    settings: Settings, dataset: str | None = None, source: str = "events"
) -> dict[str, Any]:
    """Reference = training window; current = streamed events (fallback: test window)."""
    dataset = dataset or settings.dataset
    prepared = load_prepared(settings, dataset)
    reference = prepared.split.train
    current = pd.DataFrame()
    source_used = "test_split"
    if source == "events":
        events = read_events(settings.events_dir)
        if not events.empty:
            current = events.rename(columns={"event_ts": "timestamp"})[
                ["user_id", "item_id", "rating", "timestamp"]
            ]
            source_used = f"events:{settings.events_dir}"
        else:
            logger.warning(
                "no streamed events under %s; using the test window", settings.events_dir
            )
    if current.empty:
        current = prepared.split.test

    report = drift_report(reference, current, prepared.movies, settings.drift_psi_threshold)
    report.update(dataset=dataset, current_source=source_used)
    out_dir = settings.reports_dir / "drift"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2)
    (out_dir / "latest.json").write_text(payload, encoding="utf-8")
    Path(out_dir / f"drift_{dataset}.json").write_text(payload, encoding="utf-8")
    if settings.pushgateway_url:
        try:
            push_drift_metrics(settings.pushgateway_url, report)
        except Exception:
            logger.exception("could not push drift metrics to %s", settings.pushgateway_url)
    logger.info(
        "drift: %s (detected=%s)",
        {k: v["psi"] for k, v in report["features"].items()},
        report["drift_detected"],
    )
    return report
