import pytest

from recsys.evaluation.evaluate import (
    evaluate_ranking,
    evaluate_rating,
    evaluate_streaming_value,
    flatten_metrics,
    ground_truth,
)


def test_ground_truth_uses_threshold(prepared) -> None:
    gt = ground_truth(prepared.split.test, threshold=4.0)
    test = prepared.split.test
    for user, items in list(gt.items())[:20]:
        rows = test[(test.user_id == user) & test.item_id.isin(items)]
        assert (rows.rating >= 4.0).all()


def test_ranking_segments_partition_users(implicit_model, prepared) -> None:
    result = evaluate_ranking(implicit_model, prepared.split.test, k=10)
    assert result["warm"]["n_users"] + result["cold"]["n_users"] == result["all"]["n_users"]
    assert result["warm"]["n_users"] > 0 and result["cold"]["n_users"] > 0
    assert 0.0 < result["all"]["coverage"] <= 1.0
    for segment in ("warm", "cold", "all"):
        for key in ("precision", "recall", "ndcg", "map", "hit_rate"):
            assert 0.0 <= result[segment][key] <= 1.0


def test_rating_metrics(popularity_model, prepared) -> None:
    out = evaluate_rating(popularity_model, prepared.split.test)
    assert out["rmse"] > 0 and out["mae"] > 0
    assert out["n_ratings"] == len(prepared.split.test)


def test_streaming_value_structure_and_cold_start_gain(implicit_model, prepared) -> None:
    result = evaluate_streaming_value(implicit_model, prepared.split.test, k=10, n_context=3)
    for segment in ("warm", "cold", "all"):
        assert set(result[segment]) == {"batch_only", "streaming"}
    cold = result["cold"]
    assert cold["streaming"]["n_users"] == cold["batch_only"]["n_users"] > 0
    # with planted taste structure, folding in 3 fresh events must beat popularity
    assert cold["streaming"]["ndcg"] > cold["batch_only"]["ndcg"]


def test_flatten_metric_names() -> None:
    ranking = {
        "all": {"ndcg": 0.5, "n_users": 3, "coverage": 0.1},
        "warm": {"ndcg": 0.6, "n_users": 2},
    }
    flat = flatten_metrics(ranking, {"rmse": 0.9}, k=10)
    assert flat["ndcg_at_10"] == 0.5
    assert flat["warm_ndcg_at_10"] == 0.6
    assert flat["n_users"] == 3 and flat["warm_n_users"] == 2
    assert flat["coverage_at_10"] == pytest.approx(0.1)
    assert flat["rmse"] == 0.9
