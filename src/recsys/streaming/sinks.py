"""Offline event store: append-only parquet, Hive-partitioned by event date.

``data/events/date=2017-06-21/part-<ts>-<id>.parquet``. Downstream batch jobs (drift
monitoring, retraining, feature backfills) read it with ``read_events``; because delivery
is at-least-once, readers de-duplicate on ``event_id``.
"""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path

import pandas as pd

from recsys.streaming.events import RatingEvent

logger = logging.getLogger(__name__)


class ParquetEventSink:
    def __init__(self, root: Path, flush_events: int = 5_000, flush_seconds: float = 30.0) -> None:
        self.root = Path(root)
        self.flush_events = flush_events
        self.flush_seconds = flush_seconds
        self._buffer: list[dict] = []
        self._last_flush = time.monotonic()

    def append(self, event: RatingEvent) -> None:
        self._buffer.append(event.model_dump())

    def __len__(self) -> int:
        return len(self._buffer)

    def should_flush(self) -> bool:
        if not self._buffer:
            return False
        elapsed = time.monotonic() - self._last_flush
        return len(self._buffer) >= self.flush_events or elapsed >= self.flush_seconds

    def flush(self) -> list[Path]:
        self._last_flush = time.monotonic()
        if not self._buffer:
            return []
        frame = pd.DataFrame(self._buffer)
        self._buffer = []
        frame["date"] = pd.to_datetime(frame["event_ts"], unit="s", utc=True).dt.strftime(
            "%Y-%m-%d"
        )
        written: list[Path] = []
        stamp = int(time.time() * 1000)
        for date, part in frame.groupby("date"):
            out_dir = self.root / f"date={date}"
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / f"part-{stamp}-{uuid.uuid4().hex[:8]}.parquet"
            part.drop(columns="date").to_parquet(path, index=False)
            written.append(path)
        logger.info("flushed %d events into %d files", len(frame), len(written))
        return written


def read_events(root: Path, since_ts: int | None = None) -> pd.DataFrame:
    """Load (and de-duplicate) every event under ``root``; empty frame if none."""
    files = sorted(Path(root).glob("date=*/*.parquet"))
    if not files:
        return pd.DataFrame(columns=["event_id", "user_id", "item_id", "rating", "event_ts"])
    frame = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    frame = frame.drop_duplicates("event_id", keep="first")
    if since_ts is not None:
        frame = frame[frame["event_ts"] >= since_ts]
    return frame.reset_index(drop=True)
