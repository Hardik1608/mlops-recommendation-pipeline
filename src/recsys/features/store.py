"""Online feature-store abstraction.

The serving path and the stream processor depend only on ``FeatureStoreClient``:

* ``InMemoryFeatureStore`` - tests and single-process local runs
* ``FeastFeatureStore``    - Feast SDK (sqlite online store locally, Redis in docker/k8s)
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from recsys.config import FeatureBackend, Settings


@dataclass
class UserRealtimeFeatures:
    """Last-N interactions of a user, ordered by event time (oldest first)."""

    recent_items: list[int] = field(default_factory=list)
    recent_ratings: list[float] = field(default_factory=list)
    recent_ts: list[int] = field(default_factory=list)
    event_count: int = 0
    last_event_ts: int = 0

    def after(self, cutoff_ts: int) -> tuple[list[int], list[float]]:
        """Interactions newer than ``cutoff_ts`` (i.e. not yet seen by the model)."""
        pairs = [
            (item, rating)
            for item, rating, ts in zip(
                self.recent_items, self.recent_ratings, self.recent_ts, strict=True
            )
            if ts > cutoff_ts
        ]
        return [p[0] for p in pairs], [p[1] for p in pairs]


@runtime_checkable
class FeatureStoreClient(Protocol):
    def get_user_realtime(self, user_id: int) -> UserRealtimeFeatures | None: ...

    def write_user_realtime(self, rows: Mapping[int, UserRealtimeFeatures]) -> None: ...

    def ping(self) -> bool: ...


class InMemoryFeatureStore:
    def __init__(self) -> None:
        self._rows: dict[int, UserRealtimeFeatures] = {}
        self._lock = threading.Lock()

    def get_user_realtime(self, user_id: int) -> UserRealtimeFeatures | None:
        with self._lock:
            row = self._rows.get(int(user_id))
            if row is None:
                return None
            return UserRealtimeFeatures(
                list(row.recent_items),
                list(row.recent_ratings),
                list(row.recent_ts),
                row.event_count,
                row.last_event_ts,
            )

    def write_user_realtime(self, rows: Mapping[int, UserRealtimeFeatures]) -> None:
        with self._lock:
            for user_id, row in rows.items():
                self._rows[int(user_id)] = UserRealtimeFeatures(
                    list(row.recent_items),
                    list(row.recent_ratings),
                    list(row.recent_ts),
                    row.event_count,
                    row.last_event_ts,
                )

    def ping(self) -> bool:
        return True

    def __len__(self) -> int:
        return len(self._rows)


def create_feature_store(settings: Settings) -> FeatureStoreClient:
    if settings.feature_backend == FeatureBackend.FEAST:
        from recsys.features.feast_store import FeastFeatureStore

        return FeastFeatureStore(settings.feast_repo_path, settings.feast_yaml)
    return InMemoryFeatureStore()
