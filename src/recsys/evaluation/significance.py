"""Paired bootstrap over users: is model A's NDCG@K really different from model B's?

Both models are scored on the same test users, so the comparison resamples *users* (with
replacement) and recomputes both means on each resample. That keeps the pairing, which a
comparison of two independent confidence intervals would throw away.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def paired_bootstrap(
    a: dict[int, float],
    b: dict[int, float],
    n_boot: int = 2000,
    seed: int = 0,
    users: Iterable[int] | None = None,
) -> dict[str, float | int | list[float]]:
    """Relative lift of A over B in mean NDCG, with a 95% percentile interval.

    ``users`` restricts the comparison (e.g. to returning users: cold users get the same
    popularity fallback from every model, so they only dilute the difference).
    ``p_value`` is the two-sided bootstrap p-value of "mean(A) = mean(B)": twice the share
    of resamples on the far side of zero (floored at one resample).
    """
    common = set(a) & set(b)
    if users is not None:
        common &= set(users)
    users = sorted(common)
    if not users:
        raise ValueError("no common users")
    x = np.fromiter((a[u] for u in users), dtype=np.float64, count=len(users))
    y = np.fromiter((b[u] for u in users), dtype=np.float64, count=len(users))
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    ratios = np.empty(n_boot)
    chunk = max(1, 2_000_000 // len(users))  # bound the (chunk x users) index matrix
    for start in range(0, n_boot, chunk):
        stop = min(start + chunk, n_boot)
        idx = rng.integers(0, len(users), size=(stop - start, len(users)))
        mx, my = x[idx].mean(axis=1), y[idx].mean(axis=1)
        diffs[start:stop] = mx - my
        ratios[start:stop] = mx / np.maximum(my, 1e-12) - 1.0
    tail = min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {
        "n_users": len(users),
        "mean_a": float(x.mean()),
        "mean_b": float(y.mean()),
        "relative_lift": float(x.mean() / max(y.mean(), 1e-12) - 1.0),
        "relative_lift_ci95": [
            float(np.percentile(ratios, 2.5)),
            float(np.percentile(ratios, 97.5)),
        ],
        "diff_ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
        "p_value": float(min(1.0, max(2 * tail, 1.0 / n_boot))),
        "n_boot": n_boot,
    }
