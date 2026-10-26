"""Metric definitions (binary relevance).

For a ranked list ``rec`` (length <= K) and a set of relevant items ``rel``:

* Precision@K = |rec[:K] & rel| / K
* Recall@K    = |rec[:K] & rel| / |rel|
* AP@K        = 1/min(|rel|, K) * sum_j P@j * 1[rec_j in rel]     (MAP@K = mean over users)
* NDCG@K      = DCG/IDCG, DCG = sum_j 1[rec_j in rel] / log2(j + 1)
* HitRate@K   = 1[|rec[:K] & rel| > 0]
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _discounts(n: int) -> np.ndarray:
    return 1.0 / np.log2(np.arange(2, n + 2))


def ranking_metrics(recommended: Sequence[int], relevant: set[int], k: int) -> dict[str, float]:
    """All ranking metrics for one user in a single pass."""
    if not relevant:
        raise ValueError("relevant set must not be empty")
    rec = list(recommended)[:k]
    hits = np.fromiter((item in relevant for item in rec), dtype=bool, count=len(rec))
    n_hits = int(hits.sum())
    ideal = min(len(relevant), k)
    discounts = _discounts(max(len(rec), ideal))
    dcg = float(discounts[: len(rec)][hits].sum())
    idcg = float(discounts[:ideal].sum())
    cum_hits = np.cumsum(hits)
    precision_at_j = cum_hits / np.arange(1, len(rec) + 1)
    ap = float(precision_at_j[hits].sum() / ideal) if len(rec) else 0.0
    return {
        "precision": n_hits / k,
        "recall": n_hits / len(relevant),
        "ndcg": dcg / idcg if idcg > 0 else 0.0,
        "map": ap,
        "hit_rate": float(n_hits > 0),
    }


def precision_at_k(recommended: Sequence[int], relevant: set[int], k: int) -> float:
    return ranking_metrics(recommended, relevant, k)["precision"]


def recall_at_k(recommended: Sequence[int], relevant: set[int], k: int) -> float:
    return ranking_metrics(recommended, relevant, k)["recall"]


def ndcg_at_k(recommended: Sequence[int], relevant: set[int], k: int) -> float:
    return ranking_metrics(recommended, relevant, k)["ndcg"]


def average_precision_at_k(recommended: Sequence[int], relevant: set[int], k: int) -> float:
    return ranking_metrics(recommended, relevant, k)["map"]


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))
