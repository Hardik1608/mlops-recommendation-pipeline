"""API tests against the real app, real model packages and in-memory stores."""

from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient
from tests.conftest import make_settings

from recsys.features.store import InMemoryFeatureStore, UserRealtimeFeatures
from recsys.models import ImplicitALS
from recsys.models.package import save_package
from recsys.serving.ab import assign_variant
from recsys.serving.app import create_app
from recsys.serving.cache import InMemoryTTLCache
from recsys.serving.model_loader import load_models
from recsys.serving.service import RecommendationService

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def packages(tmp_path_factory, prepared):
    root = tmp_path_factory.mktemp("packages")
    train = prepared.split.train_val
    champion = ImplicitALS(factors=8, iterations=5, regularization=1.0, alpha=5.0).fit(train)
    challenger = ImplicitALS(factors=8, iterations=5, seed=1, blend_weight=1.0).fit(train)
    save_package(champion, prepared.movies, root / "champion", {"run_id": "champ"})
    save_package(challenger, prepared.movies, root / "challenger", {"run_id": "chall"})
    return root


@pytest.fixture
def api(tmp_path, packages):
    settings = make_settings(
        tmp_path,
        champion_model_dir=packages / "champion",
        challenger_model_dir=packages / "challenger",
        challenger_traffic=0.5,
    )
    store = InMemoryFeatureStore()
    service = RecommendationService(settings, load_models(settings), store, InMemoryTTLCache(60))
    with TestClient(create_app(settings, service=service)) as client:
        yield client, service, store


def known_user(service) -> int:
    return int(service.models["champion"].package.model.user_index.ids[0])


def test_health_and_ready(api) -> None:
    client, _, _ = api
    assert client.get("/health").json() == {"status": "ok"}
    ready = client.get("/ready").json()
    assert ready["ready"] is True
    assert {m["variant"] for m in ready["models"]} == {"champion", "challenger"}
    assert ready["challenger_traffic"] == 0.5


def test_personalized_recommendations_then_cache_hit(api) -> None:
    client, service, _ = api
    user = known_user(service)
    first = client.get(f"/recommendations/{user}", params={"k": 5}).json()
    assert first["strategy"] == "personalized" and len(first["items"]) == 5
    assert first["variant"] == assign_variant(user, 0.5, service.settings.ab_salt)
    assert first["items"][0]["title"].startswith("Synthetic Movie")
    assert first["cached"] is False
    second = client.get(f"/recommendations/{user}", params={"k": 5}).json()
    assert second["cached"] is True and second["items"] == first["items"]


def test_cold_start_popularity_and_realtime_fold_in(api) -> None:
    client, service, store = api
    new_user = 10**9
    cold = client.get(f"/recommendations/{new_user}").json()
    assert cold["strategy"] == "popularity" and len(cold["items"]) == 10

    model = service.models[cold["variant"]].package
    items = model.model.item_index.ids[:3].tolist()
    after_cutoff = model.trained_through_ts + 60
    store.write_user_realtime(
        {
            new_user + 1: UserRealtimeFeatures(
                items, [5.0, 4.5, 4.0], [after_cutoff] * 3, 3, after_cutoff
            )
        }
    )
    warm = client.get(f"/recommendations/{new_user + 1}").json()
    assert warm["strategy"] == "fold_in"
    assert warm["realtime_events_used"] == 3
    assert not set(items) & {item["item_id"] for item in warm["items"]}


def test_events_older_than_training_cutoff_are_ignored(api) -> None:
    client, service, store = api
    user = 10**9 + 5
    model = service.models[service.variant_for(user)].package
    old = model.trained_through_ts - 10
    store.write_user_realtime(
        {user: UserRealtimeFeatures([int(model.model.item_index.ids[0])], [5.0], [old], 1, old)}
    )
    assert client.get(f"/recommendations/{user}").json()["strategy"] == "popularity"


def test_ab_split_across_users(api) -> None:
    client, service, _ = api
    users = service.models["champion"].package.model.user_index.ids[:60].tolist()
    variants = {
        client.get(f"/recommendations/{u}", params={"k": 3}).json()["variant"] for u in users
    }
    assert variants == {"champion", "challenger"}


@pytest.mark.parametrize(
    ("path", "params"),
    [
        ("/recommendations/1", {"k": 0}),
        ("/recommendations/1", {"k": 1000}),
        ("/recommendations/0", {}),
    ],
)
def test_input_validation(api, path, params) -> None:
    client, _, _ = api
    assert client.get(path, params=params).status_code == 422


def test_feature_store_outage_degrades_gracefully(api) -> None:
    client, service, _ = api

    class BrokenStore(InMemoryFeatureStore):
        def get_user_realtime(self, user_id):
            raise ConnectionError("online store down")

    service.feature_store = BrokenStore()
    response = client.get(f"/recommendations/{known_user(service) + 0}", params={"k": 4})
    assert response.status_code == 200
    assert response.json()["strategy"] in {"personalized", "popularity"}
    metrics = client.get("/metrics").text
    assert re.search(r"recsys_feature_store_errors_total [1-9]", metrics)


def test_prometheus_metrics_use_route_templates(api) -> None:
    client, service, _ = api
    client.get(f"/recommendations/{known_user(service)}")
    text = client.get("/metrics").text
    assert 'route="/recommendations/{user_id}"' in text
    assert "recsys_http_request_duration_seconds_bucket" in text
    assert re.search(r'recsys_model_info\{model_type="implicit_als",.*variant="champion"', text)
    assert "recsys_cache_requests_total" in text
    assert "recsys_model_compute_seconds" in text


def test_app_lifespan_loads_local_models(tmp_path, packages) -> None:
    settings = make_settings(tmp_path, champion_model_dir=packages / "champion")
    with TestClient(create_app(settings)) as client:
        ready = client.get("/ready").json()
        assert [m["variant"] for m in ready["models"]] == ["champion"]
        assert ready["challenger_traffic"] == 0.0
        assert ready["cache_backend"] == "memory"


class _ExternalStore:
    """Stand-in for a non-in-memory online store (e.g. Feast)."""

    def __init__(self) -> None:
        self.calls = 0

    def get_user_realtime(self, user_id):
        self.calls += 1
        return None

    def write_user_realtime(self, rows) -> None:
        return None

    def ping(self) -> bool:
        return True


@pytest.mark.parametrize("in_thread", [True, False])
def test_feature_lookup_threading_is_configurable(tmp_path, packages, monkeypatch, in_thread):
    import recsys.serving.service as service_module

    thread_calls: list[object] = []
    real_to_thread = service_module.asyncio.to_thread

    async def spy_to_thread(func, *args, **kwargs):
        thread_calls.append(func)
        return await real_to_thread(func, *args, **kwargs)

    monkeypatch.setattr(service_module.asyncio, "to_thread", spy_to_thread)
    settings = make_settings(
        tmp_path, champion_model_dir=packages / "champion", feature_store_in_thread=in_thread
    )
    store = _ExternalStore()
    service = RecommendationService(settings, load_models(settings), store, InMemoryTTLCache(60))
    with TestClient(create_app(settings, service=service)) as client:
        assert client.get("/recommendations/1", params={"k": 3}).status_code == 200
    assert store.calls == 1
    assert len(thread_calls) == (1 if in_thread else 0)
