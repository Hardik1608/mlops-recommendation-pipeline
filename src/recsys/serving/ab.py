"""Deterministic, sticky A/B assignment.

``bucket = sha256(salt:user_id)[:8] / 2**64`` - the same user always gets the same variant
(no flip-flopping between models mid-experiment), assignment needs no shared state, and a
new ``salt`` per experiment re-randomises users so experiments are independent.
"""

from __future__ import annotations

import hashlib

CHAMPION = "champion"
CHALLENGER = "challenger"
SHADOW = "shadow"  # scored in the background, never served


def bucket(user_id: int, salt: str) -> float:
    digest = hashlib.sha256(f"{salt}:{user_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def assign_variant(user_id: int, challenger_fraction: float, salt: str) -> str:
    if challenger_fraction <= 0.0:
        return CHAMPION
    return CHALLENGER if bucket(user_id, salt) < challenger_fraction else CHAMPION


def in_shadow_sample(user_id: int, fraction: float, salt: str) -> bool:
    """Sticky shadow sample, independent of the A/B bucket (its own salt suffix)."""
    return fraction > 0.0 and bucket(user_id, f"{salt}:shadow") < fraction
