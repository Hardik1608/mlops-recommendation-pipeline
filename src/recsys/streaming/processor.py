"""Stream processor: decode -> validate -> de-duplicate -> update online state -> sink.

Pure logic with injected dependencies, so it runs identically against Kafka, the
in-memory broker (tests, local demo) or a replay file.
"""

from __future__ import annotations

import contextlib
import logging
import signal
import time
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from recsys.features.realtime import UserActivityAggregator
from recsys.streaming.bus import EventSource
from recsys.streaming.events import EventDecodeError, decode_event
from recsys.streaming.sinks import ParquetEventSink

logger = logging.getLogger(__name__)


@dataclass
class ProcessResult:
    received: int = 0
    processed: int = 0
    duplicates: int = 0
    invalid: int = 0
    users_updated: int = 0

    def __iadd__(self, other: ProcessResult) -> ProcessResult:
        for name in ("received", "processed", "duplicates", "invalid", "users_updated"):
            setattr(self, name, getattr(self, name) + getattr(other, name))
        return self


class StreamProcessor:
    def __init__(
        self,
        aggregator: UserActivityAggregator,
        sink: ParquetEventSink | None = None,
        dead_letter: Callable[[bytes, str], None] | None = None,
        on_users_updated: Callable[[list[int]], None] | None = None,
        dedupe_capacity: int = 1_000_000,
    ) -> None:
        self.aggregator = aggregator
        self.sink = sink
        self.dead_letter = dead_letter
        self.on_users_updated = on_users_updated
        self._seen: OrderedDict[str, None] = OrderedDict()
        self._capacity = dedupe_capacity

    def _is_duplicate(self, event_id: str) -> bool:
        if event_id in self._seen:
            self._seen.move_to_end(event_id)
            return True
        self._seen[event_id] = None
        if len(self._seen) > self._capacity:
            self._seen.popitem(last=False)
        return False

    def process_batch(self, messages: Sequence[bytes]) -> ProcessResult:
        result = ProcessResult(received=len(messages))
        for raw in messages:
            try:
                event = decode_event(raw)
            except EventDecodeError as exc:
                result.invalid += 1
                if self.dead_letter is not None:
                    self.dead_letter(raw, str(exc))
                continue
            if self._is_duplicate(event.event_id):
                result.duplicates += 1
                continue
            self.aggregator.apply(event)
            if self.sink is not None:
                self.sink.append(event)
            result.processed += 1
        users = self.aggregator.flush()  # online store write: features are fresh now
        result.users_updated = len(users)
        if users and self.on_users_updated is not None:
            self.on_users_updated(users)
        return result

    def checkpoint(self) -> None:
        """Flush the offline sink. Call *before* committing broker offsets."""
        if self.sink is not None:
            self.sink.flush()


def run_consumer(
    source: EventSource,
    processor: StreamProcessor,
    *,
    max_messages: int = 500,
    poll_timeout: float = 1.0,
    commit_interval_s: float = 5.0,
    max_idle_polls: int | None = None,
) -> ProcessResult:
    """Poll -> process -> (periodically) flush sink + commit offsets, until stopped.

    ``max_idle_polls`` stops after N consecutive empty polls (used by tests / replays).
    """
    total = ProcessResult()
    stop = False

    def _stop(*_: object) -> None:
        nonlocal stop
        stop = True

    previous: dict[int, object] = {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(ValueError):  # only possible in the main thread
            previous[sig] = signal.signal(sig, _stop)

    idle, last_commit = 0, time.monotonic()
    try:
        while not stop:
            batch = source.poll_batch(max_messages=max_messages, timeout=poll_timeout)
            if batch:
                idle = 0
                total += processor.process_batch(batch)
            else:
                idle += 1
                if max_idle_polls is not None and idle >= max_idle_polls:
                    break
            due = time.monotonic() - last_commit >= commit_interval_s
            sink_due = processor.sink is not None and processor.sink.should_flush()
            if batch and (due or sink_due):
                processor.checkpoint()
                source.commit()
                last_commit = time.monotonic()
    finally:
        processor.checkpoint()
        source.commit()
        source.close()
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        logger.info("consumer stopped: %s", total)
    return total
