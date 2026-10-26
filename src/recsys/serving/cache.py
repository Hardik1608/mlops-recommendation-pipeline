"""Response cache with graceful degradation.

Layout: one Redis hash per user, ``recs:v1:<user_id>``, field ``<variant>:<model_version>:<k>``.
* the model version is part of the field, so a promotion never serves stale results;
* the stream processor invalidates a user with a single ``DEL`` when new events arrive.

If Redis is slow or down the request is served uncached. A tiny circuit breaker stops
calling Redis for ``cooldown`` seconds after an error, so an outage costs one timeout,
not one timeout per request.
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from collections.abc import Iterable
from enum import StrEnum
from typing import Protocol

from recsys.config import CacheBackend, Settings

logger = logging.getLogger(__name__)
KEY_PREFIX = "recs:v1:"


class CacheStatus(StrEnum):
    HIT = "hit"
    MISS = "miss"
    ERROR = "error"
    BYPASS = "bypass"  # circuit open / cache disabled


def cache_key(user_id: int) -> str:
    return f"{KEY_PREFIX}{user_id}"


class ResponseCache(Protocol):
    backend: str

    async def get(self, user_id: int, field: str) -> tuple[CacheStatus, bytes | None]: ...

    async def set(self, user_id: int, field: str, value: bytes) -> CacheStatus: ...

    async def invalidate(self, user_ids: Iterable[int]) -> None: ...

    async def ping(self) -> bool: ...

    async def close(self) -> None: ...


class NullCache:
    backend = "none"

    async def get(self, user_id: int, field: str) -> tuple[CacheStatus, bytes | None]:
        return CacheStatus.BYPASS, None

    async def set(self, user_id: int, field: str, value: bytes) -> CacheStatus:
        return CacheStatus.BYPASS

    async def invalidate(self, user_ids: Iterable[int]) -> None:
        return None

    async def ping(self) -> bool:
        return True

    async def close(self) -> None:
        return None


class InMemoryTTLCache:
    """Per-process LRU + TTL cache (local runs and tests)."""

    backend = "memory"

    def __init__(self, ttl_seconds: float = 300, max_entries: int = 100_000) -> None:
        self.ttl = ttl_seconds
        self.max_entries = max_entries
        self._data: OrderedDict[int, tuple[float, dict[str, bytes]]] = OrderedDict()

    async def get(self, user_id: int, field: str) -> tuple[CacheStatus, bytes | None]:
        entry = self._data.get(user_id)
        if entry is None or entry[0] < time.monotonic():
            return CacheStatus.MISS, None
        value = entry[1].get(field)
        if value is None:
            return CacheStatus.MISS, None
        self._data.move_to_end(user_id)
        return CacheStatus.HIT, value

    async def set(self, user_id: int, field: str, value: bytes) -> CacheStatus:
        expires = time.monotonic() + self.ttl
        entry = self._data.get(user_id)
        fields = entry[1] if entry is not None and entry[0] >= time.monotonic() else {}
        fields[field] = value
        self._data[user_id] = (expires, fields)
        self._data.move_to_end(user_id)
        while len(self._data) > self.max_entries:
            self._data.popitem(last=False)
        return CacheStatus.MISS

    async def invalidate(self, user_ids: Iterable[int]) -> None:
        for user_id in user_ids:
            self._data.pop(int(user_id), None)

    async def ping(self) -> bool:
        return True

    async def close(self) -> None:
        self._data.clear()


class RedisResponseCache:
    backend = "redis"

    def __init__(
        self,
        url: str,
        ttl_seconds: int = 300,
        timeout_seconds: float = 0.05,
        cooldown_seconds: float = 30.0,
    ) -> None:
        import redis.asyncio as aioredis

        self.ttl = int(ttl_seconds)
        self.cooldown = cooldown_seconds
        self._open_until = 0.0
        self._client = aioredis.from_url(
            url, socket_timeout=timeout_seconds, socket_connect_timeout=timeout_seconds
        )

    def _available(self) -> bool:
        return time.monotonic() >= self._open_until

    def _trip(self, exc: BaseException) -> None:
        if self._available():
            logger.warning("redis cache unavailable (%s); bypassing for %.0fs", exc, self.cooldown)
        self._open_until = time.monotonic() + self.cooldown

    async def get(self, user_id: int, field: str) -> tuple[CacheStatus, bytes | None]:
        if not self._available():
            return CacheStatus.BYPASS, None
        try:
            value = await self._client.hget(cache_key(user_id), field)
        except Exception as exc:  # redis errors, timeouts, connection refused
            self._trip(exc)
            return CacheStatus.ERROR, None
        return (CacheStatus.HIT, value) if value is not None else (CacheStatus.MISS, None)

    async def set(self, user_id: int, field: str, value: bytes) -> CacheStatus:
        if not self._available():
            return CacheStatus.BYPASS
        try:
            async with self._client.pipeline(transaction=False) as pipe:
                pipe.hset(cache_key(user_id), field, value)
                pipe.expire(cache_key(user_id), self.ttl)
                await pipe.execute()
        except Exception as exc:
            self._trip(exc)
            return CacheStatus.ERROR
        return CacheStatus.MISS

    async def invalidate(self, user_ids: Iterable[int]) -> None:
        keys = [cache_key(int(u)) for u in user_ids]
        if keys and self._available():
            try:
                await self._client.unlink(*keys)
            except Exception as exc:
                self._trip(exc)

    async def ping(self) -> bool:
        try:
            return bool(await self._client.ping())
        except Exception:
            return False

    async def close(self) -> None:
        await self._client.aclose()


class RedisInvalidator:
    """Synchronous invalidation used by the stream processor (one UNLINK per batch)."""

    def __init__(self, url: str, timeout_seconds: float = 0.5) -> None:
        import redis

        self._client = redis.Redis.from_url(
            url, socket_timeout=timeout_seconds, socket_connect_timeout=timeout_seconds
        )

    def __call__(self, user_ids: list[int]) -> None:
        try:
            self._client.unlink(*[cache_key(u) for u in user_ids])
        except Exception as exc:  # cache invalidation must never stop the stream
            logger.warning("cache invalidation failed: %s", exc)


def create_cache(settings: Settings) -> ResponseCache:
    if settings.cache_backend == CacheBackend.REDIS:
        return RedisResponseCache(
            settings.redis_url,
            settings.cache_ttl_seconds,
            settings.redis_timeout_seconds,
            settings.cache_cooldown_seconds,
        )
    if settings.cache_backend == CacheBackend.MEMORY:
        return InMemoryTTLCache(settings.cache_ttl_seconds, settings.cache_max_entries)
    return NullCache()
