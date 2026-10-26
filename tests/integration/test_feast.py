"""Feast end-to-end (local provider): apply, materialize, stream push, online read, API.

Runs against a temp copy of ``feature_repo/`` (SQLite online store), so it needs no services.
"""

from __future__ import annotations

import shutil

import pytest
from fastapi.testclient import TestClient

pytest.importorskip("feast")

from tests.conftest import REPO_ROOT, make_settings

from recsys.config import FeatureBackend
from recsys.features.feast_store import FeastFeatureStore
from recsys.features.offline import write_feature_tables
from recsys.features.realtime import UserActivityAggregator
from recsys.models import ImplicitALS
from recsys.models.package import save_package
from recsys.pipelines import tasks
from recsys.serving.app import create_app
from recsys.serving.cache import NullCache
from recsys.serving.model_loader import load_models
from recsys.serving.service import RecommendationService
from recsys.streaming.events import RatingEvent
from recsys.streaming.processor import StreamProcessor

pytestmark = [pytest.mark.integration, pytest.mark.slow]


@pytest.fixture(scope="module")
def feast_settings(tmp_path_factory, prepared):
    root = tmp_path_factory.mktemp("feast")
    shutil.copytree(
        REPO_ROOT / "feature_repo",
        root / "feature_repo",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    settings = make_settings(
        root, feature_backend=FeatureBackend.FEAST, feast_repo_path=root / "feature_repo"
    )
    write_feature_tables(prepared.split.train_val, prepared.movies, settings.features_dir)
    tasks.feast_apply(settings)
    tasks.feast_materialize(settings)
    return settings


@pytest.fixture(scope="module")
def store(feast_settings) -> FeastFeatureStore:
    return FeastFeatureStore(feast_settings.feast_repo_path, feast_settings.feast_yaml)


def test_batch_features_are_served_online(store, prepared) -> None:
    train = prepared.split.train_val
    user = int(train["user_id"].iloc[0])
    item = int(train["item_id"].iloc[0])
    resp = store.store.get_online_features(
        features=["user_stats:user_rating_count", "item_stats:item_rating_count"],
        entity_rows=[{"user_id": user, "item_id": item}],
    ).to_dict()
    assert resp["user_rating_count"][0] == int((train["user_id"] == user).sum())
    assert resp["item_rating_count"][0] == int((train["item_id"] == item).sum())


def test_stream_processor_pushes_realtime_features(store, prepared) -> None:
    processor = StreamProcessor(UserActivityAggregator(store, max_items=5))
    events = [
        RatingEvent.create(424242, int(item), 4.5, 2_000_000_000 + i).to_bytes()
        for i, item in enumerate(prepared.split.train_val["item_id"].unique()[:7])
    ]
    result = processor.process_batch(events)
    assert result.processed == 7
    feats = store.get_user_realtime(424242)
    assert feats is not None and len(feats.recent_items) == 5
    assert feats.event_count == 7 and feats.last_event_ts == 2_000_000_006
    # a fresh aggregator (e.g. after a consumer restart) bootstraps from the online store
    restarted = UserActivityAggregator(store, max_items=5)
    restarted.apply(RatingEvent.create(424242, 1, 3.0, 2_000_000_010))
    restarted.flush()
    assert store.get_user_realtime(424242).event_count == 8


def test_api_uses_feast_realtime_features(feast_settings, store, prepared, tmp_path) -> None:
    model = ImplicitALS(factors=8, iterations=4).fit(prepared.split.train_val)
    save_package(model, prepared.movies, tmp_path / "champion")
    settings = feast_settings.model_copy(update={"champion_model_dir": tmp_path / "champion"})
    service = RecommendationService(settings, load_models(settings), store, NullCache())
    with TestClient(create_app(settings, service=service)) as client:
        body = client.get("/recommendations/424242", params={"k": 5}).json()
        assert body["strategy"] == "fold_in"
        assert body["realtime_events_used"] == 5
        assert client.get("/ready").json()["feature_store_ok"] is True
