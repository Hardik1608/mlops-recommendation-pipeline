import pytest

from recsys.serving.ab import CHALLENGER, CHAMPION, assign_variant, bucket
from recsys.training.registry import decide


def test_assignment_is_deterministic_and_sticky() -> None:
    first = [assign_variant(u, 0.3, "exp-1") for u in range(1000)]
    second = [assign_variant(u, 0.3, "exp-1") for u in range(1000)]
    assert first == second


def test_traffic_fraction_is_respected() -> None:
    n = 20_000
    share = sum(assign_variant(u, 0.1, "exp-1") == CHALLENGER for u in range(n)) / n
    assert share == pytest.approx(0.1, abs=0.01)


def test_salt_reshuffles_users() -> None:
    a = {u for u in range(5000) if assign_variant(u, 0.5, "exp-a") == CHALLENGER}
    b = {u for u in range(5000) if assign_variant(u, 0.5, "exp-b") == CHALLENGER}
    overlap = len(a & b) / len(a)
    assert 0.4 < overlap < 0.6  # independent experiments


def test_zero_traffic_means_everyone_on_champion() -> None:
    assert {assign_variant(u, 0.0, "x") for u in range(100)} == {CHAMPION}
    assert 0.0 <= bucket(42, "x") < 1.0


def test_gate_bootstraps_first_champion() -> None:
    decision = decide("ndcg_at_10", 0.1, None)
    assert decision.action == "bootstrap_champion"
    assert decide("ndcg_at_10", 0.1, None, bootstrap_champion=False).action == "challenger"


def test_gate_requires_strict_improvement() -> None:
    assert decide("ndcg_at_10", 0.21, 0.20).action == "challenger"
    assert decide("ndcg_at_10", 0.20, 0.20).action == "rejected"
    assert decide("ndcg_at_10", 0.205, 0.20, min_relative_improvement=0.05).action == "rejected"
    assert decide("ndcg_at_10", 0.211, 0.20, min_relative_improvement=0.05).action == "challenger"


def test_gate_handles_lower_is_better_metrics() -> None:
    assert decide("rmse", 0.85, 0.90).action == "challenger"
    assert decide("rmse", 0.95, 0.90).action == "rejected"
