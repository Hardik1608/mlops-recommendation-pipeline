"""Versioned event contract for the ``ratings.v1`` topic.

Compatibility rules:
* the *major* version is in the topic name (``ratings.v1``) and in ``schema_version``;
  a breaking change ships as ``ratings.v2`` and both topics run side by side during migration;
* additive, optional fields are allowed within v1 - consumers ignore unknown fields
  (``extra="ignore"``), so old consumers keep working when producers are upgraded first.

``event_id`` is a UUIDv5 of ``(user_id, item_id, event_ts)``: replaying the same source data
produces the same ids, which makes downstream de-duplication (and therefore idempotent
at-least-once processing) possible.
"""

from __future__ import annotations

import time
import uuid
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

SCHEMA_VERSION = 1
EVENT_NAMESPACE = uuid.UUID("6f1c2a8e-3b7d-4c55-9a0e-2d8f4b1c7e93")


class EventDecodeError(ValueError):
    """Raised for payloads that cannot be parsed into a valid event (sent to the DLQ)."""


class RatingEvent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    schema_version: Literal[1] = SCHEMA_VERSION
    event_id: str = Field(min_length=1)
    event_type: Literal["rating"] = "rating"
    user_id: int = Field(gt=0)
    item_id: int = Field(gt=0)
    rating: float = Field(ge=0.5, le=5.0)
    event_ts: int = Field(gt=0, description="event time, unix seconds")
    produced_at: float | None = Field(default=None, description="producer wall clock")

    @classmethod
    def create(
        cls,
        user_id: int,
        item_id: int,
        rating: float,
        event_ts: int,
        produced_at: float | None = None,
    ) -> RatingEvent:
        event_id = str(uuid.uuid5(EVENT_NAMESPACE, f"{user_id}:{item_id}:{event_ts}"))
        return cls(
            event_id=event_id,
            user_id=int(user_id),
            item_id=int(item_id),
            rating=float(rating),
            event_ts=int(event_ts),
            produced_at=produced_at if produced_at is not None else time.time(),
        )

    @property
    def key(self) -> bytes:
        """Partition key: all events of a user land in the same partition (ordering)."""
        return str(self.user_id).encode()

    def to_bytes(self) -> bytes:
        return self.model_dump_json().encode()


def decode_event(raw: bytes | str) -> RatingEvent:
    try:
        return RatingEvent.model_validate_json(raw)
    except ValidationError as exc:
        raise EventDecodeError(str(exc.errors(include_url=False))[:500]) from exc
    except (UnicodeDecodeError, ValueError) as exc:
        raise EventDecodeError(str(exc)[:500]) from exc
