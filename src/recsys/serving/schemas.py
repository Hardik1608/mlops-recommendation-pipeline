"""API response models (they also define the OpenAPI schema)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class RecommendedItem(BaseModel):
    item_id: int
    score: float
    title: str | None = None
    genres: list[str] = Field(default_factory=list)


class RecommendationResponse(BaseModel):
    user_id: int
    k: int
    variant: str = Field(description="A/B arm: champion | challenger")
    model_version: str
    model_type: str
    strategy: str = Field(description="personalized | personalized_realtime | fold_in | popularity")
    realtime_events_used: int = 0
    cached: bool = False
    items: list[RecommendedItem]


class ModelStatus(BaseModel):
    variant: str
    model_type: str
    version: str
    source: str
    loaded_at: float
    n_users: int
    n_items: int


class ReadyResponse(BaseModel):
    ready: bool
    models: list[ModelStatus]
    cache_backend: str
    cache_ok: bool
    feature_store_ok: bool
    challenger_traffic: float
