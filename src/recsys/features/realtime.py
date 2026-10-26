"""Keyed, event-time-ordered user state maintained by the stream processor.

Kafka partitions by ``user_id``, so each user is owned by exactly one consumer: the state
can live in memory (like Flink keyed state) and is bootstrapped from the online store the
first time a user is seen after a restart or partition rebalance.
"""

from __future__ import annotations

import bisect
from collections.abc import Iterable

from recsys.features.store import FeatureStoreClient, UserRealtimeFeatures
from recsys.streaming.events import RatingEvent


class UserActivityAggregator:
    def __init__(self, store: FeatureStoreClient, max_items: int = 20) -> None:
        self.store = store
        self.max_items = max_items
        self._state: dict[int, UserRealtimeFeatures] = {}
        self._dirty: set[int] = set()

    def _get(self, user_id: int) -> UserRealtimeFeatures:
        state = self._state.get(user_id)
        if state is None:
            state = self.store.get_user_realtime(user_id) or UserRealtimeFeatures()
            self._state[user_id] = state
        return state

    def apply(self, event: RatingEvent) -> None:
        state = self._get(event.user_id)
        # A re-rating replaces the previous interaction with the same item.
        if event.item_id in state.recent_items:
            pos = state.recent_items.index(event.item_id)
            for seq in (state.recent_items, state.recent_ratings, state.recent_ts):
                del seq[pos]
        # Insert in event-time order (handles late / out-of-order events).
        pos = bisect.bisect_right(state.recent_ts, event.event_ts)
        state.recent_ts.insert(pos, event.event_ts)
        state.recent_items.insert(pos, event.item_id)
        state.recent_ratings.insert(pos, float(event.rating))
        overflow = len(state.recent_ts) - self.max_items
        if overflow > 0:  # keep the newest N
            del state.recent_ts[:overflow]
            del state.recent_items[:overflow]
            del state.recent_ratings[:overflow]
        state.event_count += 1
        state.last_event_ts = max(state.last_event_ts, event.event_ts)
        self._dirty.add(event.user_id)

    def apply_all(self, events: Iterable[RatingEvent]) -> None:
        for event in events:
            self.apply(event)

    def flush(self) -> list[int]:
        """Write every changed user to the online store; returns the flushed user ids."""
        if not self._dirty:
            return []
        users = sorted(self._dirty)
        self.store.write_user_realtime({u: self._state[u] for u in users})
        self._dirty.clear()
        return users
