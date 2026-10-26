"""Paired bootstrap used for the benchmark's model comparisons."""

from __future__ import annotations

import numpy as np
import pytest

from recsys.evaluation.significance import paired_bootstrap


def scores(n: int = 800, seed: int = 0) -> dict[int, float]:
    rng = np.random.default_rng(seed)
    return {u: float(v) for u, v in enumerate(rng.random(n))}


def test_a_real_improvement_is_detected() -> None:
    b = scores()
    a = {u: v + 0.05 for u, v in b.items()}
    res = paired_bootstrap(a, b)
    lo, hi = res["relative_lift_ci95"]
    assert 0 < lo < res["relative_lift"] < hi
    assert res["p_value"] < 0.01


def test_identical_models_are_not_significant() -> None:
    b = scores()
    res = paired_bootstrap(dict(b), b)
    assert res["relative_lift"] == 0 and res["p_value"] == 1.0


def test_noise_alone_is_usually_not_significant() -> None:
    b = scores(seed=1)
    noise = np.random.default_rng(2).normal(0, 0.05, len(b))
    a = {u: v + float(e) for (u, v), e in zip(b.items(), noise, strict=True)}
    assert paired_bootstrap(a, b)["p_value"] > 0.05


def test_only_common_users_are_compared() -> None:
    res = paired_bootstrap({1: 1.0, 2: 0.0, 3: 1.0}, {2: 0.0, 3: 0.5, 4: 1.0})
    assert res["n_users"] == 2
    with pytest.raises(ValueError):
        paired_bootstrap({1: 1.0}, {2: 1.0})


def test_comparison_can_be_restricted_to_some_users() -> None:
    a = {1: 1.0, 2: 0.0, 3: 0.5, 4: 0.5}
    b = {1: 0.0, 2: 0.0, 3: 0.5, 4: 0.5}
    everyone = paired_bootstrap(a, b)
    returning = paired_bootstrap(a, b, users=[1, 2])
    assert returning["n_users"] == 2
    assert returning["relative_lift"] > everyone["relative_lift"]  # shared users dilute it
