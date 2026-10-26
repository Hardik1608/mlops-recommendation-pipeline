import asyncio
import time

import fakeredis
import pytest
from tests.conftest import make_settings

from recsys.config import CacheBackend
from recsys.serving.cache import (
    CacheStatus,
    InMemoryTTLCache,
    NullCache,
    RedisInvalidator,
    RedisResponseCache,
    cache_key,
    create_cache,
)


def run(coro):
    return asyncio.run(coro)


def test_memory_cache_hit_miss_and_invalidate() -> None:
    cache = InMemoryTTLCache(ttl_seconds=60)
    assert run(cache.get(1, "f")) == (CacheStatus.MISS, None)
    run(cache.set(1, "f", b"payload"))
    assert run(cache.get(1, "f")) == (CacheStatus.HIT, b"payload")
    assert run(cache.get(1, "other-field"))[0] == CacheStatus.MISS
    run(cache.invalidate([1]))
    assert run(cache.get(1, "f"))[0] == CacheStatus.MISS


def test_memory_cache_ttl_and_lru_eviction(monkeypatch) -> None:
    cache = InMemoryTTLCache(ttl_seconds=10, max_entries=2)
    now = time.monotonic()
    monkeypatch.setattr(time, "monotonic", lambda: now)
    run(cache.set(1, "f", b"a"))
    run(cache.set(2, "f", b"b"))
    run(cache.set(3, "f", b"c"))  # evicts user 1 (least recently used)
    assert run(cache.get(1, "f"))[0] == CacheStatus.MISS
    monkeypatch.setattr(time, "monotonic", lambda: now + 11)
    assert run(cache.get(3, "f"))[0] == CacheStatus.MISS  # expired


def test_redis_cache_roundtrip_with_fakeredis() -> None:
    async def scenario() -> None:
        cache = RedisResponseCache("redis://unused")
        cache._client = fakeredis.FakeAsyncRedis()
        assert await cache.get(7, "champion:1:10") == (CacheStatus.MISS, None)
        assert await cache.set(7, "champion:1:10", b"x") == CacheStatus.MISS
        assert await cache.get(7, "champion:1:10") == (CacheStatus.HIT, b"x")
        assert 0 < await cache._client.ttl(cache_key(7)) <= cache.ttl
        await cache.invalidate([7])
        assert (await cache.get(7, "champion:1:10"))[0] == CacheStatus.MISS
        assert await cache.ping()

    run(scenario())


def test_redis_down_degrades_gracefully_with_circuit_breaker() -> None:
    async def scenario() -> None:
        # nothing listens on port 1: connection refused immediately
        cache = RedisResponseCache(
            "redis://127.0.0.1:1/0", timeout_seconds=0.2, cooldown_seconds=60
        )
        assert (await cache.get(1, "f"))[0] == CacheStatus.ERROR
        # breaker is open: no further network calls until the cooldown expires
        start = time.perf_counter()
        assert (await cache.get(1, "f"))[0] == CacheStatus.BYPASS
        assert await cache.set(1, "f", b"x") == CacheStatus.BYPASS
        assert time.perf_counter() - start < 0.05
        assert not await cache.ping()
        await cache.close()

    run(scenario())


def test_sync_invalidator_never_raises() -> None:
    RedisInvalidator("redis://127.0.0.1:1/0", timeout_seconds=0.2)([1, 2, 3])


@pytest.mark.parametrize(
    ("backend", "cls"),
    [(CacheBackend.MEMORY, InMemoryTTLCache), (CacheBackend.NONE, NullCache)],
)
def test_factory(tmp_path, backend, cls) -> None:
    assert isinstance(create_cache(make_settings(tmp_path, cache_backend=backend)), cls)
    assert run(NullCache().get(1, "f"))[0] == CacheStatus.BYPASS
