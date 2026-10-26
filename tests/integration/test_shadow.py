"""Shadow model: scored off the request path, measured, never served."""

from __future__ import annotations

import asyncio

import pytest
from prometheus_client import REGISTRY
from tests.conftest import make_settings

from recsys.features.store import InMemoryFeatureStore
from recsys.models import ImplicitALS
from recsys.models.package import save_package
from recsys.serving.ab import in_shadow_sample
from recsys.serving.cache import InMemoryTTLCache
from recsys.serving.model_loader import load_models
from recsys.serving.service import RecommendationService

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def packages(tmp_path_factory, prepared):
    root = tmp_path_factory.mktemp("shadow-packages")
    train = prepared.split.train_val
    champion = ImplicitALS(factors=8, iterations=5, regularization=1.0, alpha=5.0).fit(train)
    shadow = ImplicitALS(factors=8, iterations=5, seed=3, blend_weight=1.0).fit(train)
    save_package(champion, prepared.movies, root / "champion", {"run_id": "champ"})
    save_package(shadow, prepared.movies, root / "shadow", {"run_id": "shadow"})
    return root


def make_service(tmp_path, packages, shadow_traffic: float, with_shadow: bool = True):
    settings = make_settings(
        tmp_path,
        champion_model_dir=packages / "champion",
        shadow_model_dir=(packages / "shadow") if with_shadow else None,
        shadow_traffic=shadow_traffic,
        challenger_traffic=0.0,
    )
    models = load_models(settings)
    return RecommendationService(settings, models, InMemoryFeatureStore(), InMemoryTTLCache(60))


def sample(name: str, labels: dict[str, str] | None = None) -> float:
    return REGISTRY.get_sample_value(name, labels or {}) or 0.0


def recommend(service: RecommendationService, user: int, k: int = 5) -> dict:
    async def run() -> dict:
        payload = await service.recommend(user, k)
        await service.drain_shadow()
        return payload

    return asyncio.run(run())


def a_user(service: RecommendationService) -> int:
    return int(service.models["champion"].package.model.user_index.ids[1])


def test_shadow_is_scored_but_never_served(tmp_path, packages) -> None:
    service = make_service(tmp_path, packages, shadow_traffic=1.0)
    plain = make_service(tmp_path / "plain", packages, shadow_traffic=1.0, with_shadow=False)
    assert set(service.models) == {"champion", "shadow"}
    user = a_user(service)
    ok_before = sample("recsys_shadow_requests_total", {"result": "ok"})
    overlap_before = sample("recsys_shadow_overlap_ratio_count")
    timed_before = sample("recsys_model_compute_seconds_count", {"variant": "shadow"})

    payload = recommend(service, user)

    assert payload == recommend(plain, user)  # the served response ignores the shadow
    assert payload["variant"] == "champion"
    assert sample("recsys_shadow_requests_total", {"result": "ok"}) == ok_before + 1
    assert sample("recsys_shadow_overlap_ratio_count") == overlap_before + 1
    assert sample("recsys_model_compute_seconds_count", {"variant": "shadow"}) == timed_before + 1


def test_cache_hits_are_not_shadowed(tmp_path, packages) -> None:
    service = make_service(tmp_path, packages, shadow_traffic=1.0)
    user = a_user(service)
    recommend(service, user)
    before = sample("recsys_shadow_requests_total", {"result": "ok"})
    assert recommend(service, user)["cached"] is True
    assert sample("recsys_shadow_requests_total", {"result": "ok"}) == before


def test_shadow_sample_is_sticky_and_sized(tmp_path, packages) -> None:
    salt = "recsys-ab-v1"
    users = range(5_000)
    share = sum(in_shadow_sample(u, 0.2, salt) for u in users) / len(users)
    assert 0.17 < share < 0.23
    assert [in_shadow_sample(u, 0.2, salt) for u in users] == [
        in_shadow_sample(u, 0.2, salt) for u in users
    ]
    service = make_service(tmp_path, packages, shadow_traffic=0.0)
    before = sample("recsys_shadow_requests_total", {"result": "ok"})
    recommend(service, a_user(service))
    assert sample("recsys_shadow_requests_total", {"result": "ok"}) == before


def test_shadow_failures_are_counted_not_raised(tmp_path, packages, monkeypatch) -> None:
    service = make_service(tmp_path, packages, shadow_traffic=1.0)

    def broken(*args, **kwargs):
        raise RuntimeError("shadow model exploded")

    monkeypatch.setattr(service.models["shadow"].package.model, "recommend", broken)
    before = sample("recsys_shadow_requests_total", {"result": "error"})
    payload = recommend(service, a_user(service))
    assert len(payload["items"]) == 5
    assert sample("recsys_shadow_requests_total", {"result": "error"}) == before + 1
