import json

import numpy as np
import pandas as pd
import pytest

from recsys.features.offline import build_item_features, build_user_features, write_feature_tables
from recsys.monitoring.drift import (
    drift_report,
    psi_categorical,
    psi_from_counts,
    psi_numeric,
    run_drift_job,
)
from recsys.streaming.events import RatingEvent
from recsys.streaming.sinks import ParquetEventSink


def test_psi_is_zero_for_identical_distributions() -> None:
    assert psi_from_counts(np.array([10, 20, 30]), np.array([1, 2, 3])) == pytest.approx(0.0)
    rng = np.random.default_rng(0)
    assert psi_numeric(rng.normal(size=5000), rng.normal(size=5000)) < 0.02


def test_psi_detects_shift() -> None:
    rng = np.random.default_rng(0)
    assert psi_numeric(rng.normal(size=5000), rng.normal(1.0, 1.0, size=5000)) > 0.2
    ref = pd.Series(["a"] * 80 + ["b"] * 20)
    cur = pd.Series(["a"] * 20 + ["b"] * 70 + ["c"] * 10)  # new category handled
    assert psi_categorical(ref, cur) > 0.2


def test_drift_report_on_temporal_windows(prepared) -> None:
    report = drift_report(prepared.split.train, prepared.split.test, prepared.movies, 0.2)
    assert set(report["features"]) == {"rating", "item_popularity", "genre", "hour_of_day"}
    same = drift_report(prepared.split.train, prepared.split.train, prepared.movies, 0.2)
    assert all(v["psi"] < 1e-6 for v in same["features"].values())
    assert not same["drift_detected"]
    assert 0.0 <= report["new_item_share"] <= 1.0


def test_drift_job_prefers_streamed_events(settings, prepared) -> None:
    report = run_drift_job(settings, "synthetic", source="test")
    assert report["current_source"] == "test_split"
    sink = ParquetEventSink(settings.events_dir)
    for row in prepared.split.test.head(200).itertuples():
        sink.append(RatingEvent.create(row.user_id, row.item_id, row.rating, row.timestamp))
    sink.flush()
    report = run_drift_job(settings, "synthetic", source="events")
    assert report["current_source"].startswith("events:")
    assert report["n_current"] == 200
    saved = json.loads((settings.reports_dir / "drift" / "latest.json").read_text())
    assert saved["n_current"] == 200


def test_batch_features(prepared, tmp_path) -> None:
    train, movies = prepared.split.train_val, prepared.movies
    as_of = int(train["timestamp"].max())
    users = build_user_features(train, movies, as_of)
    assert users["user_id"].is_unique and len(users) == train["user_id"].nunique()
    assert (users["user_rating_count"] > 0).all()
    assert users["event_timestamp"].iloc[0] == pd.Timestamp(as_of, unit="s", tz="UTC")
    items = build_item_features(train, movies, as_of)
    assert len(items) == len(movies)
    assert items["item_popularity_rank"].min() == 1
    paths = write_feature_tables(train, movies, tmp_path)
    assert all(p.exists() for p in paths.values())
    assert pd.read_parquet(paths["user_realtime"]).empty  # schema-only push batch source
