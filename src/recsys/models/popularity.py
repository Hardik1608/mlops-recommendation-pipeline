"""Popularity baseline (ranking) + damped bias baseline (rating prediction).

Ranking: items ordered by (optionally time-decayed) interaction count in the training window.
Rating:  r_hat = mu + b_u + b_i with damped means (Koren, 2010) - the classic RMSE baseline.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from recsys.models.base import Recommender


class PopularityRecommender(Recommender):
    model_type = "popularity"

    def __init__(self, **params: Any) -> None:
        params.setdefault("item_damping", 25.0)
        params.setdefault("user_damping", 10.0)
        super().__init__(**params)
        self.item_bias = np.zeros(0, dtype=np.float32)
        self.user_bias = np.zeros(0, dtype=np.float32)

    def _fit(self, u: np.ndarray, i: np.ndarray, r: np.ndarray, ts: np.ndarray) -> None:
        mu = self.global_mean
        item_cnt = np.bincount(i, minlength=self.n_items)
        item_sum = np.bincount(i, weights=r - mu, minlength=self.n_items)
        self.item_bias = (item_sum / (item_cnt + self.params["item_damping"])).astype(np.float32)
        resid = r - mu - self.item_bias[i]
        user_cnt = np.bincount(u, minlength=self.n_users)
        user_sum = np.bincount(u, weights=resid, minlength=self.n_users)
        self.user_bias = (user_sum / (user_cnt + self.params["user_damping"])).astype(np.float32)

    def score_users(self, user_idx: np.ndarray) -> np.ndarray:
        return np.broadcast_to(self.popularity, (len(user_idx), self.n_items))

    def predict_rating(self, user_ids: np.ndarray, item_ids: np.ndarray) -> np.ndarray:
        uidx = self.user_index.lookup(user_ids)
        iidx = self.item_index.lookup(item_ids)
        pred = np.full(len(uidx), self.global_mean, dtype=np.float64)
        pred += np.where(uidx >= 0, self.user_bias[np.maximum(uidx, 0)], 0.0)
        pred += np.where(iidx >= 0, self.item_bias[np.maximum(iidx, 0)], 0.0)
        return np.clip(pred, 0.5, 5.0)

    def _arrays(self) -> dict[str, np.ndarray]:
        return {"item_bias": self.item_bias, "user_bias": self.user_bias}

    def _load_arrays(self, arrays: dict[str, np.ndarray]) -> None:
        self.item_bias = arrays["item_bias"]
        self.user_bias = arrays["user_bias"]
