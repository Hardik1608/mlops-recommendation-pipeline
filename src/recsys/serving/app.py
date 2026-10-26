"""FastAPI application factory.

Endpoints: ``GET /recommendations/{user_id}?k=10``, ``/health`` (liveness), ``/ready``
(readiness: a champion model is loaded), ``/metrics`` (Prometheus).

Run: ``uvicorn recsys.serving.app:create_app --factory`` or ``recsys serve``. One worker per pod -
scale horizontally (Kubernetes HPA) instead of multi-worker processes, which keeps
Prometheus metrics simple (no multiprocess mode) and memory per pod predictable.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Path, Query, Request
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from threadpoolctl import threadpool_limits

import recsys
from recsys.config import ModelSource, Settings
from recsys.features.store import create_feature_store
from recsys.serving import metrics as m
from recsys.serving.cache import create_cache
from recsys.serving.model_loader import load_mlflow, load_models, resolve_mlflow_versions
from recsys.serving.schemas import ModelStatus, ReadyResponse, RecommendationResponse
from recsys.serving.service import RecommendationService

logger = logging.getLogger(__name__)


class PrometheusMiddleware:
    """Pure ASGI middleware (cheaper than BaseHTTPMiddleware): latency + counts per route."""

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        start = time.perf_counter()
        status = 500

        async def send_wrapper(message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            route = getattr(scope.get("route"), "path", None) or "unmatched"
            variant = scope.get("state", {}).get("variant", "none")
            m.HTTP_REQUESTS.labels(route, scope["method"], str(status), variant).inc()
            m.HTTP_LATENCY.labels(route, variant).observe(time.perf_counter() - start)


async def _refresh_models(app: FastAPI, settings: Settings) -> None:
    """Poll registry aliases; hot-swap models when champion/challenger change."""
    while True:
        await asyncio.sleep(settings.model_refresh_seconds)
        service: RecommendationService = app.state.service
        try:
            wanted = await asyncio.to_thread(resolve_mlflow_versions, settings)
            current = {variant: lm.version for variant, lm in service.models.items()}
            if wanted and wanted != current and "champion" in wanted:
                by_version = {lm.version: lm for lm in service.models.values()}
                loaded = {}
                for variant, version in wanted.items():
                    if version in by_version:  # e.g. challenger promoted to champion
                        loaded[variant] = dataclasses.replace(by_version[version], variant=variant)
                    else:
                        loaded[variant] = await asyncio.to_thread(
                            load_mlflow, settings, variant, version
                        )
                service.set_models(loaded)
                logger.info("hot-swapped models: %s -> %s", current, wanted)
        except Exception:
            logger.exception("model refresh failed; keeping current models")


def create_app(
    settings: Settings | None = None, service: RecommendationService | None = None
) -> FastAPI:
    """App factory. With no arguments (uvicorn ``--factory``), settings come from the env."""
    if settings is None:
        from recsys.logging_utils import configure_logging

        settings = Settings()
        configure_logging(settings.log_level, settings.log_format)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        threadpool_limits(limits=settings.blas_threads)
        svc = service
        if svc is None:
            svc = RecommendationService(
                settings,
                await asyncio.to_thread(load_models, settings),
                create_feature_store(settings),
                create_cache(settings),
            )
        app.state.service = svc
        refresher = None
        if settings.model_source == ModelSource.MLFLOW and settings.model_refresh_seconds > 0:
            refresher = asyncio.create_task(_refresh_models(app, settings))
        try:
            yield
        finally:
            if refresher is not None:
                refresher.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await refresher
            await svc.drain_shadow()
            await svc.cache.close()

    app = FastAPI(
        title="Recommendation API",
        version=recsys.__version__,
        description="Top-K movie recommendations with A/B routing and real-time features.",
        lifespan=lifespan,
    )
    app.add_middleware(PrometheusMiddleware)

    @app.get("/health", tags=["ops"])
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready", response_model=ReadyResponse, tags=["ops"])
    async def ready(request: Request) -> ReadyResponse:
        svc: RecommendationService | None = getattr(request.app.state, "service", None)
        if svc is None:
            raise HTTPException(status_code=503, detail="models not loaded")
        feature_ok = await asyncio.to_thread(svc.feature_store.ping)
        return ReadyResponse(
            ready=True,
            models=[
                ModelStatus(
                    variant=lm.variant,
                    model_type=lm.model_type,
                    version=lm.version,
                    source=lm.source,
                    loaded_at=lm.loaded_at,
                    n_users=lm.package.model.n_users,
                    n_items=lm.package.model.n_items,
                )
                for lm in svc.models.values()
            ],
            cache_backend=svc.cache.backend,
            cache_ok=await svc.cache.ping(),
            feature_store_ok=feature_ok,
            challenger_traffic=svc.challenger_fraction,
        )

    @app.get("/metrics", include_in_schema=False)
    async def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get(
        "/recommendations/{user_id}",
        response_model=RecommendationResponse,
        tags=["recommendations"],
    )
    async def recommendations(
        request: Request,
        user_id: int = Path(ge=1, description="MovieLens userId (unknown ids get cold-start)"),
        k: int = Query(default=settings.default_k, ge=1, le=settings.max_k),
    ) -> dict:
        svc: RecommendationService = request.app.state.service
        variant = svc.variant_for(user_id)
        request.state.variant = variant  # picked up by the metrics middleware
        return await svc.recommend(user_id, k, variant)

    return app
