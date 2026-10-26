"""Replay ratings as live ``ratings.v1`` events (simulates production traffic).

By default the *test window* of the temporal split is replayed - interactions that happen
after the model's training cutoff, i.e. exactly what a live system would receive next.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator

import pandas as pd

from recsys.streaming.bus import EventPublisher
from recsys.streaming.events import RatingEvent

logger = logging.getLogger(__name__)


def iter_events(ratings: pd.DataFrame) -> Iterator[RatingEvent]:
    ordered = ratings.sort_values(["timestamp", "user_id"], kind="stable")
    for row in ordered.itertuples(index=False):
        yield RatingEvent.create(row.user_id, row.item_id, row.rating, row.timestamp)


def replay(
    publisher: EventPublisher,
    topic: str,
    events: Iterator[RatingEvent],
    *,
    rate: float | None = None,
    limit: int | None = None,
    log_every: int = 10_000,
) -> int:
    """Publish events (optionally rate-limited to ``rate`` events/s); returns the count."""
    sent = 0
    start = time.perf_counter()
    for event in events:
        if limit is not None and sent >= limit:
            break
        if rate:
            target = start + sent / rate
            delay = target - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
        publisher.publish(topic, event.key, event.to_bytes())
        sent += 1
        if log_every and sent % log_every == 0:
            logger.info(
                "published %d events (%.0f ev/s)", sent, sent / (time.perf_counter() - start)
            )
    publisher.flush()
    logger.info("replay finished: %d events in %.1fs", sent, time.perf_counter() - start)
    return sent
