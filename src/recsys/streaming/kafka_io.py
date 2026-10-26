"""confluent-kafka adapters for ``EventPublisher`` / ``EventSource``.

Producer: idempotent (``enable.idempotence``), ``acks=all``, small ``linger.ms`` batching.
Consumer: manual commits only (``enable.auto.commit=false``) - offsets are committed after
the online store write *and* the offline sink flush, giving at-least-once delivery; the
processor de-duplicates on the deterministic ``event_id``.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class KafkaEventPublisher:
    def __init__(self, bootstrap_servers: str, client_id: str = "recsys-producer") -> None:
        from confluent_kafka import Producer

        self._producer = Producer(
            {
                "bootstrap.servers": bootstrap_servers,
                "client.id": client_id,
                "enable.idempotence": True,
                "acks": "all",
                "linger.ms": 5,
                "compression.type": "lz4",
            }
        )
        self.delivery_errors = 0

    def _on_delivery(self, err, msg) -> None:
        if err is not None:
            self.delivery_errors += 1
            logger.error("delivery failed for key=%s: %s", msg.key(), err)

    def publish(self, topic: str, key: bytes, value: bytes) -> None:
        while True:
            try:
                self._producer.produce(topic, key=key, value=value, on_delivery=self._on_delivery)
                break
            except BufferError:  # local queue full: serve delivery callbacks, then retry
                self._producer.poll(0.5)
        self._producer.poll(0)

    def flush(self, timeout: float = 30.0) -> None:
        remaining = self._producer.flush(timeout)
        if remaining:
            raise TimeoutError(f"{remaining} messages not delivered within {timeout}s")


class KafkaEventSource:
    def __init__(self, bootstrap_servers: str, topic: str, group_id: str) -> None:
        from confluent_kafka import Consumer

        self._consumer = Consumer(
            {
                "bootstrap.servers": bootstrap_servers,
                "group.id": group_id,
                "enable.auto.commit": False,
                "auto.offset.reset": "earliest",
                "partition.assignment.strategy": "cooperative-sticky",
            }
        )
        self._consumer.subscribe([topic])

    def poll_batch(self, max_messages: int = 500, timeout: float = 1.0) -> list[bytes]:
        out: list[bytes] = []
        for msg in self._consumer.consume(num_messages=max_messages, timeout=timeout):
            if msg.error():
                logger.warning("consumer error: %s", msg.error())
                continue
            out.append(msg.value())
        return out

    def commit(self) -> None:
        try:
            self._consumer.commit(asynchronous=False)
        except Exception as exc:  # e.g. nothing to commit yet
            logger.debug("commit skipped: %s", exc)

    def close(self) -> None:
        self._consumer.close()
