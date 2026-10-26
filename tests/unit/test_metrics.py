import math

import numpy as np
import pytest

from recsys.evaluation.metrics import (
    average_precision_at_k,
    mae,
    ndcg_at_k,
    precision_at_k,
    ranking_metrics,
    recall_at_k,
    rmse,
)


def test_hand_computed_example() -> None:
    recommended = [10, 20, 30, 40]
    relevant = {20, 40, 99}
    k = 4
    assert precision_at_k(recommended, relevant, k) == pytest.approx(2 / 4)
    assert recall_at_k(recommended, relevant, k) == pytest.approx(2 / 3)
    # hits at ranks 2 and 4: AP = (1/2 + 2/4) / min(3, 4)
    assert average_precision_at_k(recommended, relevant, k) == pytest.approx((0.5 + 0.5) / 3)
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    idcg = 1 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)
    assert ndcg_at_k(recommended, relevant, k) == pytest.approx(dcg / idcg)


def test_perfect_ranking_scores_one() -> None:
    metrics = ranking_metrics([1, 2, 3], {1, 2, 3}, k=3)
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["ndcg"] == pytest.approx(1.0)
    assert metrics["map"] == pytest.approx(1.0)
    assert metrics["hit_rate"] == 1.0


def test_no_hits_scores_zero() -> None:
    metrics = ranking_metrics([4, 5, 6], {1}, k=3)
    assert all(metrics[key] == 0.0 for key in ("precision", "recall", "ndcg", "map", "hit_rate"))


def test_short_list_and_large_k() -> None:
    # fewer recommendations than K: precision still divides by K
    assert precision_at_k([1], {1}, k=10) == pytest.approx(0.1)
    assert ndcg_at_k([1], {1}, k=2000) == pytest.approx(1.0)


def test_empty_relevant_set_rejected() -> None:
    with pytest.raises(ValueError):
        ranking_metrics([1, 2], set(), k=2)


def test_rmse_mae() -> None:
    y = np.array([1.0, 2.0, 3.0])
    p = np.array([1.0, 2.0, 5.0])
    assert rmse(y, p) == pytest.approx(math.sqrt(4 / 3))
    assert mae(y, p) == pytest.approx(2 / 3)
