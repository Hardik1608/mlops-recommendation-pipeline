"""Common recommender machinery: id mapping, seen-item masking, top-k, persistence."""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar, Self

import numpy as np
import pandas as pd
import scipy.sparse as sp

FORMAT_VERSION = 1
DAY_SECONDS = 86_400


class IdIndex:
    """Bidirectional mapping between raw ids (e.g. MovieLens ``movieId``) and row indices."""

    def __init__(self, ids: Sequence[int] | np.ndarray) -> None:
        self.ids = np.asarray(ids, dtype=np.int64)
        self._lookup: dict[int, int] = {int(x): i for i, x in enumerate(self.ids.tolist())}
        self._pd_index = pd.Index(self.ids)

    def __len__(self) -> int:
        return len(self.ids)

    def __contains__(self, raw_id: object) -> bool:
        return raw_id in self._lookup

    def get(self, raw_id: int) -> int | None:
        return self._lookup.get(int(raw_id))

    def lookup(self, raw_ids: Iterable[int] | np.ndarray) -> np.ndarray:
        """Vectorised raw-id -> index; unknown ids map to ``-1``."""
        arr = raw_ids if isinstance(raw_ids, np.ndarray) else np.fromiter(raw_ids, dtype=np.int64)
        if arr.size == 0 or len(self.ids) == 0:
            return np.full(arr.size, -1, dtype=np.int64)
        return self._pd_index.get_indexer(arr.astype(np.int64))


@dataclass
class Recommendation:
    items: list[int]
    scores: list[float]
    strategy: str  # personalized | personalized_realtime | fold_in | popularity


def top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the ``k`` largest finite scores, sorted descending."""
    n = scores.shape[0]
    if k <= 0 or n == 0:
        return np.empty(0, dtype=np.int64)
    candidates = np.argpartition(-scores, k - 1)[:k] if k < n else np.arange(n)
    order = candidates[np.argsort(-scores[candidates], kind="stable")]
    return order[np.isfinite(scores[order])]


class Recommender(ABC):
    """Base class. Sub-classes implement ``_fit`` and ``score_users`` (+ optional fold-in).

    Public methods take/return *raw* ids; everything internal works on contiguous indices.
    ``seen`` is the users x items CSR matrix of training ratings: used to never
    re-recommend consumed items and as the history for real-time fold-in.
    """

    model_type: ClassVar[str] = "base"

    def __init__(self, **params: Any) -> None:
        self.params: dict[str, Any] = dict(params)
        self.user_index = IdIndex([])
        self.item_index = IdIndex([])
        self.seen: sp.csr_matrix = sp.csr_matrix((0, 0), dtype=np.float32)
        self.popularity = np.zeros(0, dtype=np.float32)
        self._popular_order = np.zeros(0, dtype=np.int64)
        self.global_mean = 0.0
        self.metadata: dict[str, Any] = {}

    # ------------------------------------------------------------------ fitting
    def fit(self, ratings: pd.DataFrame) -> Self:
        if ratings.empty:
            raise ValueError("cannot fit on an empty ratings frame")
        start = time.perf_counter()
        ratings = ratings.sort_values("timestamp", kind="stable").drop_duplicates(
            ["user_id", "item_id"], keep="last"
        )
        self.user_index = IdIndex(np.unique(ratings["user_id"].to_numpy()))
        self.item_index = IdIndex(np.unique(ratings["item_id"].to_numpy()))
        u = self.user_index.lookup(ratings["user_id"].to_numpy())
        i = self.item_index.lookup(ratings["item_id"].to_numpy())
        r = ratings["rating"].to_numpy(dtype=np.float64)
        ts = ratings["timestamp"].to_numpy(dtype=np.int64)

        self.seen = sp.csr_matrix(
            (r.astype(np.float32), (u, i)), shape=(self.n_users, self.n_items)
        )
        self.global_mean = float(r.mean())
        self.popularity = self._popularity(i, ts)
        self._popular_order = np.argsort(-self.popularity, kind="stable")

        self._fit(u, i, r, ts)

        self.metadata.update(
            model_type=self.model_type,
            n_users=self.n_users,
            n_items=self.n_items,
            n_ratings=len(ratings),
            data_max_timestamp=int(ts.max()),
            fit_seconds=round(time.perf_counter() - start, 3),
            trained_at=datetime.now(tz=UTC).isoformat(),
        )
        return self

    def _popularity(self, item_idx: np.ndarray, ts: np.ndarray) -> np.ndarray:
        half_life = self.params.get("popularity_half_life_days")
        weights = None
        if half_life:
            age_days = (ts.max() - ts) / DAY_SECONDS
            weights = np.power(0.5, age_days / float(half_life))
        return np.bincount(item_idx, weights=weights, minlength=self.n_items).astype(np.float32)

    @abstractmethod
    def _fit(self, u: np.ndarray, i: np.ndarray, r: np.ndarray, ts: np.ndarray) -> None: ...

    # ------------------------------------------------------------------ scoring
    @property
    def n_users(self) -> int:
        return len(self.user_index)

    @property
    def n_items(self) -> int:
        return len(self.item_index)

    @abstractmethod
    def score_users(self, user_idx: np.ndarray) -> np.ndarray:
        """Return a ``(len(user_idx), n_items)`` float32 matrix of ranking scores."""

    def score_context(self, item_idx: np.ndarray, ratings: np.ndarray) -> np.ndarray | None:
        """Scores for a user described by interactions (fold-in); ``None`` if unsupported."""
        return None

    def predict_rating(self, user_ids: np.ndarray, item_ids: np.ndarray) -> np.ndarray | None:
        """Predicted explicit ratings, or ``None`` for models that only rank (implicit)."""
        return None

    # ------------------------------------------------------------------ raw-id API
    def knows_user(self, user_id: int) -> bool:
        return int(user_id) in self.user_index

    def _known(self, raw_ids: Iterable[int]) -> np.ndarray:
        idx = self.item_index.lookup(raw_ids)
        return idx[idx >= 0]

    def _history(self, uidx: int) -> tuple[np.ndarray, np.ndarray]:
        start, end = self.seen.indptr[uidx], self.seen.indptr[uidx + 1]
        return self.seen.indices[start:end], self.seen.data[start:end].astype(np.float64)

    def _context_history(self, uidx: int) -> tuple[np.ndarray, np.ndarray]:
        """History passed to ``score_context`` before fresh events. Sequence models override
        this to return the interactions in time order (``seen`` is ordered by item)."""
        return self._history(uidx)

    def recommend(
        self,
        user_id: int,
        k: int = 10,
        exclude: Iterable[int] = (),
        context_items: Sequence[int] = (),
        context_ratings: Sequence[float] | None = None,
    ) -> Recommendation:
        """Top-k for one user, optionally using fresh (streamed) interactions as context.

        * known user, no context      -> batch embedding           (``personalized``)
        * known user + context         -> fold-in history + context (``personalized_realtime``)
        * unknown user + context       -> fold-in context only      (``fold_in``)
        * otherwise                    -> most popular              (``popularity``)
        """
        ctx_raw = np.asarray(list(context_items), dtype=np.int64)
        ctx_r = np.asarray(
            context_ratings if context_ratings is not None else [4.0] * len(ctx_raw),
            dtype=np.float64,
        )
        ctx_idx = self.item_index.lookup(ctx_raw)
        ctx_valid = ctx_idx >= 0
        ctx_idx, ctx_r = ctx_idx[ctx_valid], ctx_r[ctx_valid]
        uidx = self.user_index.get(user_id)

        scores: np.ndarray | None = None
        strategy = "popularity"
        mask = [self._known(exclude), ctx_idx]
        if uidx is not None:
            hist_idx, _ = self._history(uidx)
            mask.append(hist_idx)
            if len(ctx_idx):
                seq_idx, seq_r = self._context_history(uidx)
                scores = self.score_context(
                    np.concatenate([seq_idx, ctx_idx]), np.concatenate([seq_r, ctx_r])
                )
                strategy = "personalized_realtime"
            if scores is None:
                scores = self.score_users(np.array([uidx]))[0]
                strategy = "personalized"
        elif len(ctx_idx):
            scores = self.score_context(ctx_idx, ctx_r)
            strategy = "fold_in"
        if scores is None:
            return self.recommend_popular(k, [*exclude, *ctx_raw.tolist()])

        scores = np.array(scores, dtype=np.float32, copy=True)
        scores[np.concatenate(mask)] = -np.inf
        top = top_k_indices(scores, k)
        return Recommendation(
            items=self.item_index.ids[top].tolist(),
            scores=scores[top].astype(float).tolist(),
            strategy=strategy,
        )

    def recommend_popular(self, k: int = 10, exclude: Iterable[int] = ()) -> Recommendation:
        excluded = set(self._known(exclude).tolist())
        top: list[int] = []
        for idx in self._popular_order[: k + len(excluded)].tolist():
            if idx not in excluded:
                top.append(idx)
                if len(top) == k:
                    break
        arr = np.asarray(top, dtype=np.int64)
        return Recommendation(
            items=self.item_index.ids[arr].tolist(),
            scores=self.popularity[arr].astype(float).tolist(),
            strategy="popularity",
        )

    def recommend_batch(self, user_idx: np.ndarray, k: int) -> np.ndarray:
        """Vectorised top-k item *indices* for known users (offline evaluation)."""
        scores = np.array(self.score_users(user_idx), dtype=np.float32, copy=True)
        rows = self.seen[user_idx]
        row_ids = np.repeat(np.arange(len(user_idx)), np.diff(rows.indptr))
        scores[row_ids, rows.indices] = -np.inf
        kk = min(k, self.n_items)
        part = np.argpartition(-scores, kk - 1, axis=1)[:, :kk]
        order = np.argsort(-np.take_along_axis(scores, part, axis=1), axis=1, kind="stable")
        return np.take_along_axis(part, order, axis=1)

    # ------------------------------------------------------------------ persistence
    def _arrays(self) -> dict[str, np.ndarray]:
        return {}

    def _load_arrays(self, arrays: dict[str, np.ndarray]) -> None:
        return None

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        arrays = {
            "user_ids": self.user_index.ids,
            "item_ids": self.item_index.ids,
            "popularity": self.popularity,
            "seen_indptr": self.seen.indptr,
            "seen_indices": self.seen.indices,
            "seen_data": self.seen.data.astype(np.float32),
            **self._arrays(),
        }
        np.savez_compressed(path / "arrays.npz", **arrays)
        meta = {
            "format_version": FORMAT_VERSION,
            "model_type": self.model_type,
            "params": self.params,
            "global_mean": self.global_mean,
            "metadata": self.metadata,
        }
        (path / "model.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
        return path

    @classmethod
    def from_saved(cls, meta: dict[str, Any], arrays: dict[str, np.ndarray]) -> Self:
        model = cls(**meta.get("params", {}))
        model.user_index = IdIndex(arrays["user_ids"])
        model.item_index = IdIndex(arrays["item_ids"])
        model.popularity = arrays["popularity"].astype(np.float32)
        model._popular_order = np.argsort(-model.popularity, kind="stable")
        model.seen = sp.csr_matrix(
            (arrays["seen_data"], arrays["seen_indices"], arrays["seen_indptr"]),
            shape=(len(model.user_index), len(model.item_index)),
        )
        model.global_mean = float(meta.get("global_mean", 0.0))
        model.metadata = dict(meta.get("metadata", {}))
        model._load_arrays(arrays)
        return model
