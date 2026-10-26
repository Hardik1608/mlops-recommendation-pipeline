"""Feast-backed online feature store (push for writes, ``get_online_features`` for reads)."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from recsys.features.store import UserRealtimeFeatures

logger = logging.getLogger(__name__)

REALTIME_VIEW = "user_realtime"
PUSH_SOURCE = "user_realtime_push"
REALTIME_FIELDS = (
    "recent_items",
    "recent_ratings",
    "recent_ts",
    "rt_event_count",
    "last_event_ts",
)


class FeastFeatureStore:
    def __init__(self, repo_path: Path, yaml_name: str = "feature_store.yaml") -> None:
        from feast import FeatureStore

        repo_path = Path(repo_path).resolve()
        self._store = FeatureStore(repo_path=str(repo_path), fs_yaml_file=repo_path / yaml_name)
        self._refs = [f"{REALTIME_VIEW}:{name}" for name in REALTIME_FIELDS]

    @property
    def store(self):
        return self._store

    def get_user_realtime(self, user_id: int) -> UserRealtimeFeatures | None:
        resp = self._store.get_online_features(
            features=self._refs, entity_rows=[{"user_id": int(user_id)}]
        ).to_dict()
        items = resp.get("recent_items", [None])[0]
        if items is None:
            return None
        return UserRealtimeFeatures(
            recent_items=[int(x) for x in items],
            recent_ratings=[float(x) for x in (resp["recent_ratings"][0] or [])],
            recent_ts=[int(x) for x in (resp["recent_ts"][0] or [])],
            event_count=int(resp["rt_event_count"][0] or 0),
            last_event_ts=int(resp["last_event_ts"][0] or 0),
        )

    def write_user_realtime(self, rows: Mapping[int, UserRealtimeFeatures]) -> None:
        if not rows:
            return
        from feast.data_source import PushMode

        now = pd.Timestamp(datetime.now(tz=UTC))
        frame = pd.DataFrame(
            [
                {
                    "user_id": int(user_id),
                    # Feast keeps the row with the newest event_timestamp per entity.
                    "event_timestamp": pd.Timestamp(row.last_event_ts, unit="s", tz="UTC"),
                    "created": now,
                    "recent_items": [int(x) for x in row.recent_items],
                    "recent_ratings": [float(x) for x in row.recent_ratings],
                    "recent_ts": [int(x) for x in row.recent_ts],
                    "rt_event_count": int(row.event_count),
                    "last_event_ts": int(row.last_event_ts),
                }
                for user_id, row in rows.items()
            ]
        )
        self._store.push(PUSH_SOURCE, frame, to=PushMode.ONLINE)

    def ping(self) -> bool:
        try:
            self._store.get_feature_view(REALTIME_VIEW)
            return True
        except Exception:  # pragma: no cover - depends on registry availability
            logger.exception("Feast registry not reachable")
            return False
