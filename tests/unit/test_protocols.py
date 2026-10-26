"""Leave-one-out split used by the benchmark's protocol check."""

from __future__ import annotations

import pandas as pd

from recsys.evaluation.protocols import leave_one_out_split


def test_leave_one_out_holds_out_each_users_last_two_events() -> None:
    ratings = pd.DataFrame(
        {
            "user_id": [1, 1, 1, 1, 2, 2, 3, 3, 3],
            "item_id": [10, 11, 12, 13, 20, 21, 30, 31, 32],
            "rating": [4.0, 3.0, 5.0, 1.0, 4.0, 4.0, 2.0, 5.0, 3.0],
            # user 1's events are shuffled in the frame; order must come from time
            "timestamp": [40, 10, 30, 20, 5, 6, 100, 300, 200],
        }
    )
    train, val, test = leave_one_out_split(ratings)
    assert dict(zip(test["user_id"], test["item_id"], strict=True)) == {1: 10, 3: 31}
    assert dict(zip(val["user_id"], val["item_id"], strict=True)) == {1: 12, 3: 32}
    assert sorted(train[train["user_id"] == 2]["item_id"]) == [20, 21]  # < 3 events: all kept
    assert sorted(train[train["user_id"] == 1]["item_id"]) == [11, 13]
    assert len(train) + len(val) + len(test) == len(ratings)
    held_out = set(zip(val["user_id"], val["item_id"], strict=True)) | set(
        zip(test["user_id"], test["item_id"], strict=True)
    )
    assert held_out.isdisjoint(zip(train["user_id"], train["item_id"], strict=True))
