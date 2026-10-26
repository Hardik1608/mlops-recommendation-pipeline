"""Broker abstractions + an in-memory partitioned log.

``InMemoryBroker`` mimics the Kafka semantics the processor relies on - key-based
partitioning, per-group positions and explicit commits - so producer/consumer logic,
including redelivery after a crash, is testable without a broker.
"""

from __future__ import annotations

import threading
import zlib
from collections import defaultdict
from dataclasses import dataclass
from typing import Protocol


class EventPublisher(Protocol):
    def publish(self, topic: str, key: bytes, value: bytes) -> None: ...

    def flush(self, timeout: float = 30.0) -> None: ...


class EventSource(Protocol):
    def poll_batch(self, max_messages: int = 500, timeout: float = 1.0) -> list[bytes]: ...

    def commit(self) -> None: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class Record:
    key: bytes
    value: bytes


class InMemoryBroker:
    def __init__(self, partitions: int = 3) -> None:
        self.partitions = partitions
        self._logs: dict[str, list[list[Record]]] = defaultdict(
            lambda: [[] for _ in range(self.partitions)]
        )
        self._committed: dict[tuple[str, str], list[int]] = {}
        self._lock = threading.Lock()

    # -- producer side ----------------------------------------------------------------
    def publish(self, topic: str, key: bytes, value: bytes) -> None:
        partition = zlib.crc32(key) % self.partitions
        with self._lock:
            self._logs[topic][partition].append(Record(key, value))

    def flush(self, timeout: float = 30.0) -> None:
        return None

    def size(self, topic: str) -> int:
        return sum(len(p) for p in self._logs[topic])

    def messages(self, topic: str) -> list[Record]:
        return [rec for part in self._logs[topic] for rec in part]

    # -- consumer side ----------------------------------------------------------------
    def consumer(self, topic: str, group: str) -> InMemoryConsumer:
        return InMemoryConsumer(self, topic, group)

    def committed(self, topic: str, group: str) -> list[int]:
        return list(self._committed.get((topic, group), [0] * self.partitions))


class InMemoryConsumer:
    """Starts from the group's committed offsets, like a restarted Kafka consumer."""

    def __init__(self, broker: InMemoryBroker, topic: str, group: str) -> None:
        self.broker, self.topic, self.group = broker, topic, group
        self._position = broker.committed(topic, group)

    def poll_batch(self, max_messages: int = 500, timeout: float = 0.0) -> list[bytes]:
        out: list[bytes] = []
        with self.broker._lock:
            logs = self.broker._logs[self.topic]
            for partition, log in enumerate(logs):
                start = self._position[partition]
                take = log[start : start + (max_messages - len(out))]
                out.extend(rec.value for rec in take)
                self._position[partition] += len(take)
                if len(out) >= max_messages:
                    break
        return out

    def commit(self) -> None:
        with self.broker._lock:
            self.broker._committed[(self.topic, self.group)] = list(self._position)

    def close(self) -> None:
        return None
