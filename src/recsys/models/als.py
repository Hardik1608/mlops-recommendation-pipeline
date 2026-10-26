"""Alternating Least Squares matrix factorisation in plain numpy/scipy.

* ``ExplicitALS``  - biased MF for rating prediction (minimises squared error; ALS-WR
  regularisation as in Zhou et al. 2008 / Spark MLlib). Good RMSE.
* ``ImplicitALS``  - confidence-weighted MF for implicit feedback (Hu, Koren & Volinsky
  2008). Every rating is an interaction with confidence ``1 + alpha * w``; ``w`` can be
  the rating and can decay with age (``recency_half_life_days``). Good ranking.

Both support *fold-in*: computing an embedding for a user from a handful of streamed
interactions with one ``k x k`` solve - real-time personalisation without retraining.

Performance: rows are sorted by interaction count and processed in padded batches, which
removes per-row Python overhead. Implicit ALS uses the conjugate-gradient solver by default
(matrix-free, O(nnz * k) per step); ``solver="exact"`` keeps the closed-form normal-equation
solve as a reference (the tests check both agree). ``backend="torch"`` runs the same solvers
on the GPU in float32 (:mod:`recsys.models.als_torch`); the default is numpy (float64).
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Any

import numpy as np
import scipy.sparse as sp

from recsys.models.base import DAY_SECONDS, Recommender

PAIR_CHUNK = 1_000_000  # (user, item) pairs scored at once: bounds the gathered factors


def pair_scores(a: np.ndarray, b: np.ndarray, rows_a: np.ndarray, rows_b: np.ndarray) -> np.ndarray:
    """``sum(a[rows_a] * b[rows_b], axis=1)`` without materialising all gathered rows at once
    (20M ratings x 64 factors would be ~10 GB per side)."""
    out = np.empty(len(rows_a), dtype=np.float64)
    for start in range(0, len(rows_a), PAIR_CHUNK):
        part = slice(start, start + PAIR_CHUNK)
        out[part] = np.einsum("ij,ij->i", a[rows_a[part]], b[rows_b[part]])
    return out


def _csr(rows: np.ndarray, cols: np.ndarray, values: np.ndarray, shape: tuple[int, int]):
    matrix = sp.csr_matrix((values, (rows, cols)), shape=shape, dtype=np.float64)
    matrix.sum_duplicates()
    return matrix


def row_batches(
    csr: sp.csr_matrix, width: int, max_rows: int = 2048, budget: int = 2_000_000
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Yield ``(rows, pos, valid)`` for padded batches of non-empty CSR rows.

    Rows are sorted by length so padding waste is small; the batch size adapts so a
    gathered ``(batch, longest_row, width)`` tensor stays under ``budget`` floats.
    ``pos`` indexes ``csr.indices``/``csr.data``; ``valid`` masks the padding.
    """
    counts = np.diff(csr.indptr)
    order = np.argsort(counts, kind="stable")
    order = order[counts[order] > 0]
    i, n = 0, len(order)
    while i < n:
        j = min(i + max_rows, n)
        longest = int(counts[order[j - 1]])
        size = max(1, min(j - i, budget // max(longest * width, 1)))
        rows = order[i : i + size]
        lens = counts[rows]
        span = np.arange(int(lens.max()))
        valid = span[None, :] < lens[:, None]
        pos = csr.indptr[rows][:, None] + np.where(valid, span[None, :], 0)
        yield rows, pos, valid
        i += size


class _ALSBase(Recommender):
    def __init__(self, **params: Any) -> None:
        params.setdefault("factors", 64)
        params.setdefault("regularization", 0.1)
        params.setdefault("iterations", 15)
        params.setdefault("seed", 42)
        params.setdefault("backend", "numpy")  # numpy (CPU, float64) | torch (GPU, float32)
        params.setdefault("device", "auto")  # torch backend only
        super().__init__(**params)
        self.user_factors = np.zeros((0, 0), dtype=np.float32)
        self.item_factors = np.zeros((0, 0), dtype=np.float32)
        self.history: list[dict[str, float]] = []

    @property
    def k(self) -> int:
        return int(self.params["factors"])

    @property
    def reg(self) -> float:
        return float(self.params["regularization"])

    def _arrays(self) -> dict[str, np.ndarray]:
        return {"user_factors": self.user_factors, "item_factors": self.item_factors}

    def _load_arrays(self, arrays: dict[str, np.ndarray]) -> None:
        self.user_factors = arrays["user_factors"].astype(np.float32)
        self.item_factors = arrays["item_factors"].astype(np.float32)


class ImplicitALS(_ALSBase):
    model_type = "implicit_als"

    def __init__(self, **params: Any) -> None:
        params.setdefault("alpha", 10.0)
        params.setdefault("weighting", "rating")  # rating | binary
        params.setdefault("recency_half_life_days", None)
        params.setdefault("solver", "cg")  # cg (Takacs et al. 2011) | exact
        params.setdefault("cg_steps", 3)
        params.setdefault("blend_weight", 0.0)  # beta: weight of the trending-popularity prior
        params.setdefault("blend_half_life_days", None)  # recency of that prior ("trending")
        super().__init__(**params)
        self._gram: np.ndarray | None = None
        self._prior: np.ndarray | None = None
        self.blend_popularity = np.zeros(0, dtype=np.float32)

    def _confidence_minus_one(self, ratings: np.ndarray) -> np.ndarray:
        ratings = np.asarray(ratings, dtype=np.float64)
        weights = ratings if self.params["weighting"] == "rating" else np.ones_like(ratings)
        return float(self.params["alpha"]) * weights

    def _fit(self, u: np.ndarray, i: np.ndarray, r: np.ndarray, ts: np.ndarray) -> None:
        rng = np.random.default_rng(self.params["seed"])
        c1 = self._confidence_minus_one(r)
        half_life = self.params.get("recency_half_life_days")
        if half_life:
            c1 = c1 * np.power(0.5, (ts.max() - ts) / DAY_SECONDS / float(half_life))
        blend_half_life = self.params.get("blend_half_life_days")
        weights = (
            np.power(0.5, (ts.max() - ts) / DAY_SECONDS / float(blend_half_life))
            if blend_half_life
            else None
        )
        self.blend_popularity = np.bincount(i, weights=weights, minlength=self.n_items).astype(
            np.float32
        )
        cui = _csr(u, i, c1, (self.n_users, self.n_items))
        x = rng.normal(0.0, 0.01, size=(self.n_users, self.k))
        y = rng.normal(0.0, 0.01, size=(self.n_items, self.k))
        if self.params.get("backend") == "torch":
            from recsys.models import als_torch

            x, y, self.history = als_torch.fit_implicit(cui, x, y, self.params)
            self.user_factors, self.item_factors = x.astype(np.float32), y.astype(np.float32)
            self._gram = None
            self._prior = None
            return
        ciu = cui.T.tocsr()
        use_cg = self.params["solver"] == "cg"
        steps = int(self.params["cg_steps"])
        self.history = []
        for it in range(int(self.params["iterations"])):
            start = time.perf_counter()
            if use_cg:
                x = self.solve_cg(y, cui, self.reg, x, steps)
                y = self.solve_cg(x, ciu, self.reg, y, steps)
            else:
                x = self.solve(y, cui, self.reg)
                y = self.solve(x, ciu, self.reg)
            self.history.append({"iteration": it + 1, "seconds": time.perf_counter() - start})
        self.user_factors = x.astype(np.float32)
        self.item_factors = y.astype(np.float32)
        self._gram = None
        self._prior = None

    @staticmethod
    def solve_cg(
        fixed: np.ndarray, conf: sp.csr_matrix, reg: float, x0: np.ndarray, steps: int = 3
    ) -> np.ndarray:
        """Approximate the exact solve with a few warm-started conjugate-gradient steps.

        A x = (Y'Y + reg*I) x + Y_u' diag(c_u - 1) Y_u x is applied matrix-free, so the
        cost is O(nnz * k) per step instead of O(nnz * k^2 + rows * k^3).
        """
        k = fixed.shape[1]
        gram = fixed.T @ fixed + reg * np.eye(k)
        out = x0.copy()
        for rows, pos, valid in row_batches(conf, k, budget=4_000_000):
            f = fixed[conf.indices[pos]]  # (b, L, k)
            c1 = np.where(valid, conf.data[pos], 0.0)  # (b, L); padding contributes 0

            def apply_a(v: np.ndarray, f: np.ndarray = f, c1: np.ndarray = c1) -> np.ndarray:
                return v @ gram + np.einsum("blk,bl->bk", f, c1 * np.einsum("blk,bk->bl", f, v))

            x = out[rows]
            rhs = np.einsum("blk,bl->bk", f, np.where(valid, 1.0 + c1, 0.0))
            r = rhs - apply_a(x)
            p = r.copy()
            rs = np.einsum("bk,bk->b", r, r)
            for _ in range(steps):
                ap = apply_a(p)
                denom = np.einsum("bk,bk->b", p, ap)
                step = np.divide(rs, denom, out=np.zeros_like(rs), where=denom > 1e-20)
                x = x + step[:, None] * p
                r = r - step[:, None] * ap
                rs_new = np.einsum("bk,bk->b", r, r)
                beta = np.divide(rs_new, rs, out=np.zeros_like(rs), where=rs > 1e-20)
                p = r + beta[:, None] * p
                rs = rs_new
            out[rows] = x
        return out

    @staticmethod
    def solve(fixed: np.ndarray, conf: sp.csr_matrix, reg: float) -> np.ndarray:
        """x_u = (Y'Y + Y'(C_u - I)Y + reg*I)^-1 Y' C_u p_u for every row of ``conf``."""
        k = fixed.shape[1]
        gram = fixed.T @ fixed + reg * np.eye(k)
        out = np.zeros((conf.shape[0], k))
        for rows, pos, valid in row_batches(conf, k):
            f = fixed[conf.indices[pos]]  # (b, L, k)
            c1 = np.where(valid, conf.data[pos], 0.0)  # (b, L); padding contributes 0
            a = gram + np.matmul(f.transpose(0, 2, 1) * c1[:, None, :], f)
            rhs = np.einsum("blk,bl->bk", f, np.where(valid, 1.0 + c1, 0.0))
            out[rows] = np.linalg.solve(a, rhs[..., None])[..., 0]
        return out

    def _blend(self, scores: np.ndarray) -> np.ndarray:
        """Hybrid score = per-user max-normalised ALS score + beta * log-popularity prior.

        The prior is the item popularity of the training window, time-decayed with
        ``blend_half_life_days`` ("trending now"): under a temporal split this is a strong
        signal that pure MF misses (see README results). It is independent of the
        cold-start fallback ranking, which uses ``popularity_half_life_days``.
        """
        beta = float(self.params.get("blend_weight") or 0.0)
        if beta <= 0.0:
            return scores
        if self._prior is None:
            pop = self.blend_popularity.astype(np.float64)
            self._prior = (beta * np.log1p(pop) / np.log1p(max(pop.max(), 1e-9))).astype(np.float32)
        scale = np.maximum(np.abs(scores).max(axis=-1, keepdims=True), 1e-9)
        return scores / scale + self._prior

    def score_users(self, user_idx: np.ndarray) -> np.ndarray:
        return self._blend(self.user_factors[user_idx] @ self.item_factors.T)

    def fold_in(self, item_idx: np.ndarray, ratings: np.ndarray) -> np.ndarray:
        if self._gram is None:
            y64 = self.item_factors.astype(np.float64)
            self._gram = y64.T @ y64 + self.reg * np.eye(self.k)
        f = self.item_factors[item_idx].astype(np.float64)
        c1 = self._confidence_minus_one(ratings)
        a = self._gram + (f.T * c1) @ f
        return np.linalg.solve(a, f.T @ (1.0 + c1))

    def score_context(self, item_idx: np.ndarray, ratings: np.ndarray) -> np.ndarray:
        return self._blend(self.item_factors @ self.fold_in(item_idx, ratings).astype(np.float32))

    def _arrays(self) -> dict[str, np.ndarray]:
        return {**super()._arrays(), "blend_popularity": self.blend_popularity}

    def _load_arrays(self, arrays: dict[str, np.ndarray]) -> None:
        super()._load_arrays(arrays)
        self.blend_popularity = arrays["blend_popularity"].astype(np.float32)


class ExplicitALS(_ALSBase):
    model_type = "explicit_als"

    def __init__(self, **params: Any) -> None:
        params.setdefault("factors", 32)
        params.setdefault("regularization", 0.05)
        super().__init__(**params)
        self.user_bias = np.zeros(0, dtype=np.float32)
        self.item_bias = np.zeros(0, dtype=np.float32)

    def _fit(self, u: np.ndarray, i: np.ndarray, r: np.ndarray, ts: np.ndarray) -> None:
        rng = np.random.default_rng(self.params["seed"])
        mu = self.global_mean
        rui = _csr(u, i, r, (self.n_users, self.n_items))
        riu = rui.T.tocsr()
        x = rng.normal(0.0, 0.1, size=(self.n_users, self.k))
        y = rng.normal(0.0, 0.1, size=(self.n_items, self.k))
        if self.params.get("backend") == "torch":
            from recsys.models import als_torch

            x, y, bu, bi, self.history = als_torch.fit_explicit(rui, u, i, r, mu, x, y, self.params)
            self.user_factors, self.item_factors = x.astype(np.float32), y.astype(np.float32)
            self.user_bias, self.item_bias = bu.astype(np.float32), bi.astype(np.float32)
            return
        bu = np.zeros(self.n_users)
        bi = np.zeros(self.n_items)
        self.history = []
        for it in range(int(self.params["iterations"])):
            start = time.perf_counter()
            x, bu = self.solve(y, bi, rui, mu, self.reg)
            y, bi = self.solve(x, bu, riu, mu, self.reg)
            pred = mu + bu[u] + bi[i] + pair_scores(x, y, u, i)
            rmse = float(np.sqrt(np.mean((r - pred) ** 2)))
            self.history.append(
                {"iteration": it + 1, "train_rmse": rmse, "seconds": time.perf_counter() - start}
            )
        self.user_factors = x.astype(np.float32)
        self.item_factors = y.astype(np.float32)
        self.user_bias = bu.astype(np.float32)
        self.item_bias = bi.astype(np.float32)

    @staticmethod
    def solve(
        fixed: np.ndarray, fixed_bias: np.ndarray, ratings: sp.csr_matrix, mu: float, reg: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Solve [x_u, b_u] for each row with ALS-WR regularisation (reg * n_u)."""
        k = fixed.shape[1]
        aug = np.hstack([fixed, np.ones((fixed.shape[0], 1))])
        eye = np.eye(k + 1)
        factors = np.zeros((ratings.shape[0], k))
        bias = np.zeros(ratings.shape[0])
        for rows, pos, valid in row_batches(ratings, k + 1):
            idx = ratings.indices[pos]
            z = aug[idx] * valid[..., None]  # padded rows become zero vectors
            target = np.where(valid, ratings.data[pos] - mu - fixed_bias[idx], 0.0)
            n = valid.sum(axis=1)
            a = np.matmul(z.transpose(0, 2, 1), z) + reg * n[:, None, None] * eye
            w = np.linalg.solve(a, np.einsum("blk,bl->bk", z, target)[..., None])[..., 0]
            factors[rows] = w[:, :k]
            bias[rows] = w[:, k]
        return factors, bias

    def score_users(self, user_idx: np.ndarray) -> np.ndarray:
        # mu + b_u is constant per user, so it does not change the ranking.
        return self.user_factors[user_idx] @ self.item_factors.T + self.item_bias

    def predict_rating(self, user_ids: np.ndarray, item_ids: np.ndarray) -> np.ndarray:
        uidx = self.user_index.lookup(user_ids)
        iidx = self.item_index.lookup(item_ids)
        ku, ki = uidx >= 0, iidx >= 0
        pred = np.full(len(uidx), self.global_mean, dtype=np.float64)
        pred[ku] += self.user_bias[uidx[ku]]
        pred[ki] += self.item_bias[iidx[ki]]
        both = ku & ki
        pred[both] += pair_scores(self.user_factors, self.item_factors, uidx[both], iidx[both])
        return np.clip(pred, 0.5, 5.0)

    def fold_in(self, item_idx: np.ndarray, ratings: np.ndarray) -> tuple[np.ndarray, float]:
        z = np.hstack([self.item_factors[item_idx], np.ones((len(item_idx), 1))]).astype(np.float64)
        target = np.asarray(ratings, dtype=np.float64) - self.global_mean - self.item_bias[item_idx]
        a = z.T @ z + self.reg * len(item_idx) * np.eye(self.k + 1)
        w = np.linalg.solve(a, z.T @ target)
        return w[: self.k], float(w[self.k])

    def score_context(self, item_idx: np.ndarray, ratings: np.ndarray) -> np.ndarray:
        x, _ = self.fold_in(item_idx, ratings)
        return self.item_factors @ x.astype(np.float32) + self.item_bias

    def _arrays(self) -> dict[str, np.ndarray]:
        return {**super()._arrays(), "user_bias": self.user_bias, "item_bias": self.item_bias}

    def _load_arrays(self, arrays: dict[str, np.ndarray]) -> None:
        super()._load_arrays(arrays)
        self.user_bias = arrays["user_bias"].astype(np.float32)
        self.item_bias = arrays["item_bias"].astype(np.float32)
