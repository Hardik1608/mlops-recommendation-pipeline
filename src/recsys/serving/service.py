"""Request path: A/B assign -> cache -> real-time features -> model -> cache fill.

Every dependency failure degrades instead of failing the request:
cache down -> uncached; feature store down -> no real-time context; unknown user ->
fold-in from streamed events, else popularity.

Shadow model (optional): for a sticky sample of uncached requests, the shadow model scores
the same user and real-time context in a background thread *after* the response is built.
Its latency and its overlap with the served list go to Prometheus; its output is dropped.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time

from recsys.config import Settings
from recsys.features.store import FeatureStoreClient, InMemoryFeatureStore, UserRealtimeFeatures
from recsys.models.base import Recommendation
from recsys.serving import metrics as m
from recsys.serving.ab import CHALLENGER, CHAMPION, SHADOW, assign_variant, in_shadow_sample
from recsys.serving.cache import CacheStatus, ResponseCache
from recsys.serving.model_loader import LoadedModel

logger = logging.getLogger(__name__)


class RecommendationService:
    def __init__(
        self,
        settings: Settings,
        models: dict[str, LoadedModel],
        feature_store: FeatureStoreClient,
        cache: ResponseCache,
    ) -> None:
        self.settings = settings
        self.feature_store = feature_store
        self.cache = cache
        self.models: dict[str, LoadedModel] = {}
        self._shadow_tasks: set[asyncio.Task] = set()  # keep references until done
        self.set_models(models)

    # ------------------------------------------------------------------ model management
    def set_models(self, models: dict[str, LoadedModel]) -> None:
        """Atomically swap the served models (used at startup and by the hot-swap poller)."""
        if CHAMPION not in models:
            raise ValueError("a champion model is required")
        self.models = dict(models)
        m.MODEL_INFO.clear()
        for variant, loaded in self.models.items():
            m.MODEL_INFO.labels(variant, loaded.model_type, loaded.version).set(1)
            m.MODEL_LOADED_AT.labels(variant).set(loaded.loaded_at)

    @property
    def challenger_fraction(self) -> float:
        return self.settings.challenger_traffic if CHALLENGER in self.models else 0.0

    def variant_for(self, user_id: int) -> str:
        return assign_variant(user_id, self.challenger_fraction, self.settings.ab_salt)

    # ------------------------------------------------------------------ request path
    async def _realtime(self, user_id: int) -> UserRealtimeFeatures | None:
        try:
            inline = isinstance(self.feature_store, InMemoryFeatureStore) or (
                not self.settings.feature_store_in_thread
            )
            if inline:
                return self.feature_store.get_user_realtime(user_id)
            # Feast's SDK is synchronous: keep network waits (Redis) off the event loop.
            return await asyncio.to_thread(self.feature_store.get_user_realtime, user_id)
        except Exception:
            m.FEATURE_STORE_ERRORS.inc()
            logger.warning("feature store lookup failed for user %s", user_id, exc_info=True)
            return None

    async def recommend(self, user_id: int, k: int, variant: str | None = None) -> dict:
        variant = variant or self.variant_for(user_id)
        loaded = self.models.get(variant)
        if loaded is None:  # challenger removed by a hot-swap since the variant was chosen
            variant, loaded = CHAMPION, self.models[CHAMPION]
        field = f"{variant}:{loaded.version}:{k}"

        status, cached = await self.cache.get(user_id, field)
        m.CACHE_REQUESTS.labels(status.value).inc()
        if cached is not None:
            payload = json.loads(cached)
            payload["cached"] = True
            m.RECOMMENDATIONS.labels(variant, payload["strategy"]).inc()
            return payload

        realtime = await self._realtime(user_id)
        context_items: list[int] = []
        context_ratings: list[float] = []
        if realtime is not None:
            context_items, context_ratings = realtime.after(loaded.package.trained_through_ts)

        start = time.perf_counter()
        rec = loaded.package.model.recommend(
            user_id, k, context_items=context_items, context_ratings=context_ratings
        )
        m.MODEL_COMPUTE.labels(variant).observe(time.perf_counter() - start)
        if rec.strategy in ("fold_in", "popularity"):
            m.COLD_STARTS.labels(rec.strategy).inc()
        m.RECOMMENDATIONS.labels(variant, rec.strategy).inc()

        catalog = loaded.package.catalog
        items = []
        for item_id, score in zip(rec.items, rec.scores, strict=True):
            title, genres = catalog.get(item_id, (None, ""))
            items.append(
                {
                    "item_id": item_id,
                    "score": round(score, 6),
                    "title": title,
                    "genres": genres.split("|") if genres else [],
                }
            )
        payload = {
            "user_id": user_id,
            "k": k,
            "variant": variant,
            "model_version": loaded.version,
            "model_type": loaded.model_type,
            "strategy": rec.strategy,
            "realtime_events_used": len(context_items),
            "cached": False,
            "items": items,
        }
        status = await self.cache.set(user_id, field, json.dumps(payload).encode())
        if status == CacheStatus.ERROR:
            m.CACHE_REQUESTS.labels("set_error").inc()
        if SHADOW in self.models and in_shadow_sample(
            user_id, self.settings.shadow_traffic, self.settings.ab_salt
        ):
            task = asyncio.create_task(self._shadow(user_id, k, realtime, rec.items))
            self._shadow_tasks.add(task)
            task.add_done_callback(self._shadow_tasks.discard)
        return payload

    async def _shadow(
        self, user_id: int, k: int, realtime: UserRealtimeFeatures | None, served: list[int]
    ) -> None:
        """Score the shadow model off the request path; never raises."""
        shadow = self.models.get(SHADOW)
        if shadow is None:  # removed by a hot-swap meanwhile
            return
        context_items: list[int] = []
        context_ratings: list[float] = []
        if realtime is not None:  # each model sees the events newer than *its* training data
            context_items, context_ratings = realtime.after(shadow.package.trained_through_ts)

        def score() -> tuple[Recommendation, float]:
            start = time.perf_counter()
            rec = shadow.package.model.recommend(
                user_id, k, context_items=context_items, context_ratings=context_ratings
            )
            return rec, time.perf_counter() - start

        try:
            rec, seconds = await asyncio.to_thread(score)
        except Exception:
            m.SHADOW_REQUESTS.labels("error").inc()
            logger.warning("shadow scoring failed for user %s", user_id, exc_info=True)
            return
        m.SHADOW_REQUESTS.labels("ok").inc()
        m.MODEL_COMPUTE.labels(SHADOW).observe(seconds)
        m.SHADOW_OVERLAP.observe(len(set(rec.items) & set(served)) / max(k, 1))

    async def drain_shadow(self) -> None:
        """Wait for in-flight shadow scoring (tests and graceful shutdown)."""
        if self._shadow_tasks:
            await asyncio.gather(*self._shadow_tasks, return_exceptions=True)
