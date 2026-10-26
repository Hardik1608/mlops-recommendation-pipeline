"""GPU backend for ALS training (PyTorch, float32).

The same algorithms as the numpy solvers in :mod:`recsys.models.als` - conjugate gradient
for implicit ALS, the ALS-WR closed form for explicit ALS - over the same padded row
batches, starting from the same initial factors. Only training moves to the GPU: the factors
come back as numpy arrays, so saved models, evaluation and serving are unchanged. Large
catalogues stay in GPU memory instead of competing for host RAM.

Imported only when a model is configured with ``backend: torch``.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import scipy.sparse as sp
import torch

from recsys.models.als import row_batches
from recsys.models.sasrec_torch import resolve_device

GPU_BUDGET = 32_000_000  # floats in one gathered (rows x longest x k) batch (128 MB)


class _Rows:
    """One CSR matrix (users x items or items x users) and its padded batches on a device."""

    def __init__(self, csr: sp.csr_matrix, width: int, device: torch.device) -> None:
        self.shape = csr.shape
        self.indices = torch.from_numpy(csr.indices.astype(np.int64)).to(device)
        self.data = torch.from_numpy(csr.data.astype(np.float32)).to(device)
        self.batches = [
            (
                torch.from_numpy(rows.astype(np.int64)).to(device),
                torch.from_numpy(pos.astype(np.int64)).to(device),
                torch.from_numpy(valid).to(device),
            )
            for rows, pos, valid in row_batches(csr, width, max_rows=65_536, budget=GPU_BUDGET)
        ]


def _safe_div(num: torch.Tensor, den: torch.Tensor) -> torch.Tensor:
    ok = den > 1e-20
    return torch.where(ok, num / torch.where(ok, den, torch.ones_like(den)), torch.zeros_like(num))


def _implicit_cg(
    fixed: torch.Tensor, conf: _Rows, reg: float, x0: torch.Tensor, steps: int
) -> torch.Tensor:
    """Warm-started CG for every row: (Y'Y + reg I + Y_u' C1_u Y_u) x_u = Y_u' (1 + c1_u)."""
    k = fixed.shape[1]
    gram = fixed.T @ fixed + reg * torch.eye(k, device=fixed.device)
    out = x0.clone()
    for rows, pos, valid in conf.batches:
        f = fixed[conf.indices[pos]]  # (b, L, k)
        c1 = torch.where(valid, conf.data[pos], torch.zeros((), device=f.device))

        def apply_a(v: torch.Tensor, f: torch.Tensor = f, c1: torch.Tensor = c1) -> torch.Tensor:
            return v @ gram + torch.einsum("blk,bl->bk", f, c1 * torch.einsum("blk,bk->bl", f, v))

        x = out[rows]
        rhs = torch.einsum("blk,bl->bk", f, torch.where(valid, 1.0 + c1, torch.zeros_like(c1)))
        r = rhs - apply_a(x)
        p = r.clone()
        rs = (r * r).sum(dim=1)
        for _ in range(steps):
            ap = apply_a(p)
            step = _safe_div(rs, (p * ap).sum(dim=1))
            x = x + step[:, None] * p
            r = r - step[:, None] * ap
            rs_new = (r * r).sum(dim=1)
            p = r + _safe_div(rs_new, rs)[:, None] * p
            rs = rs_new
        out[rows] = x
    return out


def fit_implicit(
    cui: sp.csr_matrix,
    x0: np.ndarray,
    y0: np.ndarray,
    params: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float]]]:
    """Implicit ALS with the CG solver; ``cui`` holds confidence - 1 (users x items)."""
    if params.get("solver", "cg") != "cg":
        raise ValueError("the torch backend implements the conjugate-gradient solver only")
    device = resolve_device(str(params.get("device", "auto")))
    k, reg, steps = int(params["factors"]), float(params["regularization"]), int(params["cg_steps"])
    users = _Rows(cui, k, device)
    items = _Rows(cui.T.tocsr(), k, device)
    x = torch.from_numpy(x0.astype(np.float32)).to(device)
    y = torch.from_numpy(y0.astype(np.float32)).to(device)
    history = []
    for it in range(int(params["iterations"])):
        start = time.perf_counter()
        x = _implicit_cg(y, users, reg, x, steps)
        y = _implicit_cg(x, items, reg, y, steps)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        history.append({"iteration": it + 1, "seconds": time.perf_counter() - start})
    return x.cpu().numpy(), y.cpu().numpy(), history


def _explicit_solve(
    fixed: torch.Tensor, fixed_bias: torch.Tensor, ratings: _Rows, mu: float, reg: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """ALS-WR: solve [x_u, b_u] per row with regularisation reg * n_u."""
    k = fixed.shape[1]
    aug = torch.cat([fixed, torch.ones((fixed.shape[0], 1), device=fixed.device)], dim=1)
    eye = torch.eye(k + 1, device=fixed.device)
    factors = torch.zeros((ratings.shape[0], k), device=fixed.device)
    bias = torch.zeros(ratings.shape[0], device=fixed.device)
    for rows, pos, valid in ratings.batches:
        idx = ratings.indices[pos]
        z = aug[idx] * valid[..., None]
        zero = torch.zeros((), device=z.device)
        target = torch.where(valid, ratings.data[pos] - mu - fixed_bias[idx], zero)
        n = valid.sum(dim=1).to(z.dtype)
        a = z.transpose(1, 2) @ z + reg * n[:, None, None] * eye
        w = torch.linalg.solve(a, torch.einsum("blk,bl->bk", z, target)[..., None])[..., 0]
        factors[rows] = w[:, :k]
        bias[rows] = w[:, k]
    return factors, bias


def fit_explicit(
    rui: sp.csr_matrix,
    u: np.ndarray,
    i: np.ndarray,
    r: np.ndarray,
    mu: float,
    x0: np.ndarray,
    y0: np.ndarray,
    params: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, float]]]:
    """Biased explicit ALS; also tracks the training RMSE per iteration (in chunks)."""
    device = resolve_device(str(params.get("device", "auto")))
    k, reg = int(params["factors"]), float(params["regularization"])
    users = _Rows(rui, k + 1, device)
    items = _Rows(rui.T.tocsr(), k + 1, device)
    x = torch.from_numpy(x0.astype(np.float32)).to(device)
    y = torch.from_numpy(y0.astype(np.float32)).to(device)
    bu = torch.zeros(x.shape[0], device=device)
    bi = torch.zeros(y.shape[0], device=device)
    ut = torch.from_numpy(u.astype(np.int64)).to(device)
    it_ = torch.from_numpy(i.astype(np.int64)).to(device)
    rt = torch.from_numpy(r.astype(np.float32)).to(device)
    history = []
    for it in range(int(params["iterations"])):
        start = time.perf_counter()
        x, bu = _explicit_solve(y, bi, users, mu, reg)
        y, bi = _explicit_solve(x, bu, items, mu, reg)
        sse = torch.zeros((), device=device, dtype=torch.float64)
        for s in range(0, len(rt), 2_000_000):
            sl = slice(s, s + 2_000_000)
            pred = mu + bu[ut[sl]] + bi[it_[sl]] + (x[ut[sl]] * y[it_[sl]]).sum(dim=1)
            sse += ((rt[sl] - pred) ** 2).sum(dtype=torch.float64)
        rmse = float(torch.sqrt(sse / max(len(rt), 1)))
        history.append(
            {"iteration": it + 1, "train_rmse": rmse, "seconds": time.perf_counter() - start}
        )
    return (
        x.cpu().numpy(),
        y.cpu().numpy(),
        bu.cpu().numpy(),
        bi.cpu().numpy(),
        history,
    )
