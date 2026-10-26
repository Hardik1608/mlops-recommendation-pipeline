"""Evaluation protocol.

Ranking (Top-K): a test user is evaluated if they have >= 1 *relevant* test interaction
(rating >= threshold). Items the user rated in training are excluded from candidates.
Users are reported in three segments:

* ``warm`` - users seen in training (the model's own embedding is used)
* ``cold`` - users first seen after the training cutoff (served by the popularity fallback)
* ``all``  - what the serving system actually delivers

Rating (RMSE/MAE): every test rating, with the model's fallback for unknown users/items.

Streaming value: for each test user, the first ``n_context`` test events play the role of
"just-streamed" interactions. ``batch_only`` ignores them (what a nightly batch system
can do); ``streaming`` folds them into the user embedding. Metrics are computed on the
user's *remaining* relevant test interactions.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from recsys.evaluation.metrics import mae, ranking_metrics, rmse
from recsys.models.base import Recommender

RANKING_KEYS = ("precision", "recall", "ndcg", "map", "hit_rate")


def ground_truth(test: pd.DataFrame, threshold: float) -> dict[int, set[int]]:
    relevant = test[test["rating"] >= threshold]
    return {
        int(user): set(items.tolist()) for user, items in relevant.groupby("user_id")["item_id"]
    }


def _aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {key: float("nan") for key in RANKING_KEYS} | {"n_users": 0}
    return {key: float(np.mean([r[key] for r in rows])) for key in RANKING_KEYS} | {
        "n_users": len(rows)
    }


def evaluate_ranking(
    model: Recommender,
    test: pd.DataFrame,
    k: int = 10,
    threshold: float = 4.0,
    batch_size: int = 1024,
    per_user: dict[int, float] | None = None,
) -> dict[str, dict[str, float]]:
    """Segment metrics. ``per_user`` (optional) is filled with each user's NDCG@K, for
    paired significance tests between models evaluated on the same users."""
    gt = ground_truth(test, threshold)
    users = np.fromiter(gt.keys(), dtype=np.int64, count=len(gt))
    uidx = model.user_index.lookup(users)
    warm = uidx >= 0

    segments: dict[str, list[dict[str, float]]] = defaultdict(list)
    recommended: set[int] = set()

    warm_users, warm_idx = users[warm], uidx[warm]
    for start in range(0, len(warm_idx), batch_size):
        top = model.recommend_batch(warm_idx[start : start + batch_size], k)
        for user, items in zip(warm_users[start : start + batch_size], top, strict=True):
            raw = model.item_index.ids[items].tolist()
            recommended.update(raw)
            segments["warm"].append(row := ranking_metrics(raw, gt[int(user)], k))
            if per_user is not None:
                per_user[int(user)] = row["ndcg"]

    popular = model.recommend_popular(k).items
    for user in users[~warm]:
        segments["cold"].append(row := ranking_metrics(popular, gt[int(user)], k))
        if per_user is not None:
            per_user[int(user)] = row["ndcg"]
    if (~warm).any():
        recommended.update(popular)

    result = {name: _aggregate(segments[name]) for name in ("warm", "cold")}
    result["all"] = _aggregate(segments["warm"] + segments["cold"])
    result["all"]["coverage"] = len(recommended) / max(model.n_items, 1)
    return result


def evaluate_rating(model: Recommender, test: pd.DataFrame) -> dict[str, float] | None:
    pred = model.predict_rating(test["user_id"].to_numpy(), test["item_id"].to_numpy())
    if pred is None:
        return None
    y = test["rating"].to_numpy(dtype=float)
    known = (model.user_index.lookup(test["user_id"].to_numpy()) >= 0) & (
        model.item_index.lookup(test["item_id"].to_numpy()) >= 0
    )
    out = {"rmse": rmse(y, pred), "mae": mae(y, pred), "n_ratings": len(y)}
    if known.any():
        out["rmse_known_pairs"] = rmse(y[known], pred[known])
        out["n_known_pairs"] = int(known.sum())
    return out


def evaluate_streaming_value(
    model: Recommender,
    test: pd.DataFrame,
    k: int = 10,
    threshold: float = 4.0,
    n_context: int = 5,
) -> dict[str, dict[str, dict[str, float]]]:
    rows: dict[tuple[str, str], list[dict[str, float]]] = defaultdict(list)
    ordered = test.sort_values(["user_id", "timestamp"], kind="stable")
    for user, events in ordered.groupby("user_id", sort=False):
        if len(events) <= n_context:
            continue
        context, rest = events.iloc[:n_context], events.iloc[n_context:]
        relevant = set(rest.loc[rest["rating"] >= threshold, "item_id"].tolist())
        if not relevant:
            continue
        segment = "warm" if model.knows_user(int(user)) else "cold"
        batch = model.recommend(int(user), k)
        streamed = model.recommend(
            int(user),
            k,
            context_items=context["item_id"].tolist(),
            context_ratings=context["rating"].tolist(),
        )
        rows[(segment, "batch_only")].append(ranking_metrics(batch.items, relevant, k))
        rows[(segment, "streaming")].append(ranking_metrics(streamed.items, relevant, k))

    out: dict[str, dict[str, dict[str, float]]] = {}
    for segment in ("warm", "cold", "all"):
        out[segment] = {}
        for policy in ("batch_only", "streaming"):
            if segment == "all":
                merged = rows[("warm", policy)] + rows[("cold", policy)]
            else:
                merged = rows[(segment, policy)]
            out[segment][policy] = _aggregate(merged)
    return out


def flatten_metrics(
    ranking: dict[str, dict[str, float]], rating: dict[str, float] | None, k: int
) -> dict[str, float]:
    """MLflow-friendly flat metric names, e.g. ``ndcg_at_10``, ``warm_ndcg_at_10``."""
    flat: dict[str, float] = {}
    for segment, values in ranking.items():
        prefix = "" if segment == "all" else f"{segment}_"
        for key, value in values.items():
            name = f"{prefix}n_users" if key == "n_users" else f"{prefix}{key}_at_{k}"
            flat[name] = float(value)
    if rating:
        flat.update({key: float(value) for key, value in rating.items()})
    return flat
