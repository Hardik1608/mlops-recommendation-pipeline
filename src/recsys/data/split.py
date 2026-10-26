"""Global temporal split: train < val_start <= val < test_start <= test.

A *global* time cutoff (not a per-user leave-last-out) is the only split that guarantees
no information from the future leaks into training: every training event happened
strictly before every validation/test event. The price is realistic cold-start - users
who first appear after the cutoff are unknown to the model, exactly as in production.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TemporalSplit:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    val_start: int
    test_start: int

    @property
    def train_val(self) -> pd.DataFrame:
        """Train + validation, used to fit the model that is evaluated on test."""
        return pd.concat([self.train, self.val], ignore_index=True)

    def summary(self) -> dict[str, int]:
        return {
            "n_train": len(self.train),
            "n_val": len(self.val),
            "n_test": len(self.test),
            "val_start": self.val_start,
            "test_start": self.test_start,
        }


def temporal_split(
    ratings: pd.DataFrame, val_fraction: float = 0.1, test_fraction: float = 0.1
) -> TemporalSplit:
    if not 0 < test_fraction < 1 or not 0 <= val_fraction < 1:
        raise ValueError("fractions must be in [0, 1)")
    if val_fraction + test_fraction >= 1:
        raise ValueError("val_fraction + test_fraction must be < 1")

    ts = np.sort(ratings["timestamp"].to_numpy())
    n = len(ts)
    val_start = int(ts[int(np.floor(n * (1 - val_fraction - test_fraction)))])
    test_start = int(ts[int(np.floor(n * (1 - test_fraction)))])

    t = ratings["timestamp"]
    train = ratings[t < val_start].reset_index(drop=True)
    val = ratings[(t >= val_start) & (t < test_start)].reset_index(drop=True)
    test = ratings[t >= test_start].reset_index(drop=True)
    if train.empty or test.empty:
        raise ValueError("temporal split produced an empty train or test set")
    return TemporalSplit(train=train, val=val, test=test, val_start=val_start, test_start=test_start)
