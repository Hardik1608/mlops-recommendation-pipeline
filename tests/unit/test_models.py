import numpy as np
import pytest
import scipy.sparse as sp

from recsys.config import ModelConfig
from recsys.evaluation.evaluate import evaluate_ranking, evaluate_rating
from recsys.models import ExplicitALS, ImplicitALS, PopularityRecommender, build_model, load_model
from recsys.models.als import _csr
from recsys.models.base import IdIndex, top_k_indices


def _reference_implicit_solve(fixed, conf, reg):
    """Straightforward per-row normal equations - the textbook formulation."""
    k = fixed.shape[1]
    gram = fixed.T @ fixed + reg * np.eye(k)
    out = np.zeros((conf.shape[0], k))
    for row in range(conf.shape[0]):
        s, e = conf.indptr[row], conf.indptr[row + 1]
        if s == e:
            continue
        f, c1 = fixed[conf.indices[s:e]], conf.data[s:e]
        out[row] = np.linalg.solve(gram + (f.T * c1) @ f, f.T @ (1 + c1))
    return out


def _reference_explicit_solve(fixed, fixed_bias, ratings, mu, reg):
    k = fixed.shape[1]
    aug = np.hstack([fixed, np.ones((fixed.shape[0], 1))])
    factors, bias = np.zeros((ratings.shape[0], k)), np.zeros(ratings.shape[0])
    for row in range(ratings.shape[0]):
        s, e = ratings.indptr[row], ratings.indptr[row + 1]
        if s == e:
            continue
        idx = ratings.indices[s:e]
        z, t = aug[idx], ratings.data[s:e] - mu - fixed_bias[idx]
        w = np.linalg.solve(z.T @ z + reg * (e - s) * np.eye(k + 1), z.T @ t)
        factors[row], bias[row] = w[:k], w[k]
    return factors, bias


@pytest.fixture(scope="module")
def random_problem():
    rng = np.random.default_rng(0)
    n_rows, n_cols, k = 40, 60, 8
    dense = (rng.random((n_rows, n_cols)) < 0.15) * rng.integers(1, 6, (n_rows, n_cols))
    dense[0] = 0  # an empty row must be handled
    conf = sp.csr_matrix(dense.astype(float))
    return rng.normal(size=(n_cols, k)), conf


def test_batched_implicit_solve_matches_reference(random_problem) -> None:
    fixed, conf = random_problem
    np.testing.assert_allclose(
        ImplicitALS.solve(fixed, conf, 0.5), _reference_implicit_solve(fixed, conf, 0.5), atol=1e-9
    )


def test_conjugate_gradient_converges_to_exact_solution(random_problem) -> None:
    fixed, conf = random_problem
    exact = ImplicitALS.solve(fixed, conf, 0.5)
    cg = ImplicitALS.solve_cg(fixed, conf, 0.5, np.zeros_like(exact), steps=fixed.shape[1] * 2)
    np.testing.assert_allclose(cg, exact, atol=1e-6)


def test_batched_explicit_solve_matches_reference(random_problem) -> None:
    fixed, conf = random_problem
    bias = np.random.default_rng(1).normal(size=fixed.shape[0])
    got = ExplicitALS.solve(fixed, bias, conf, 3.0, 0.1)
    want = _reference_explicit_solve(fixed, bias, conf, 3.0, 0.1)
    np.testing.assert_allclose(got[0], want[0], atol=1e-9)
    np.testing.assert_allclose(got[1], want[1], atol=1e-9)


def test_id_index_roundtrip() -> None:
    index = IdIndex([101, 7, 55])
    assert index.get(7) == 1 and index.get(999) is None
    np.testing.assert_array_equal(index.lookup([55, 999, 101]), [2, -1, 0])
    assert 101 in index and len(index) == 3


def test_top_k_skips_masked_items() -> None:
    scores = np.array([0.1, 0.9, -np.inf, 0.5], dtype=np.float32)
    np.testing.assert_array_equal(top_k_indices(scores, 3), [1, 3, 0])
    np.testing.assert_array_equal(top_k_indices(scores, 10), [1, 3, 0])


def test_popularity_ranks_by_count_and_predicts_bias(popularity_model, prepared) -> None:
    counts = prepared.split.train_val["item_id"].value_counts()
    rec = popularity_model.recommend_popular(5)
    assert rec.items[0] == counts.index[0]
    assert rec.strategy == "popularity"
    pred = popularity_model.predict_rating(np.array([10**9]), np.array([counts.index[0]]))
    assert 0.5 <= pred[0] <= 5.0


def test_implicit_als_beats_popularity_on_planted_structure(
    implicit_model, popularity_model, prepared
):
    test = prepared.split.test
    als = evaluate_ranking(implicit_model, test, k=10)["warm"]["ndcg"]
    pop = evaluate_ranking(popularity_model, test, k=10)["warm"]["ndcg"]
    assert als > pop


def test_explicit_als_learns_ratings(prepared) -> None:
    model = ExplicitALS(factors=8, regularization=0.1, iterations=8).fit(prepared.split.train_val)
    losses = [step["train_rmse"] for step in model.history]
    assert losses[-1] < losses[0]
    baseline = PopularityRecommender().fit(prepared.split.train_val)
    test = prepared.split.test
    assert (
        evaluate_rating(model, test)["rmse_known_pairs"]
        < evaluate_rating(baseline, test)["rmse_known_pairs"]
    )
    assert evaluate_rating(ImplicitALS(iterations=1).fit(prepared.split.train), test) is None


def test_recommend_excludes_seen_and_requested_items(implicit_model, prepared) -> None:
    user = int(prepared.split.train_val["user_id"].iloc[0])
    seen = set(prepared.split.train_val.loc[prepared.split.train_val.user_id == user, "item_id"])
    first = implicit_model.recommend(user, k=10)
    assert first.strategy == "personalized" and len(first.items) == 10
    assert not seen & set(first.items)
    second = implicit_model.recommend(user, k=10, exclude=first.items[:3])
    assert not set(first.items[:3]) & set(second.items)


def test_cold_start_strategies(implicit_model, prepared) -> None:
    unknown_user = 10**9
    items = prepared.split.train_val["item_id"].unique()[:4].tolist()
    assert implicit_model.recommend(unknown_user, k=5).strategy == "popularity"
    folded = implicit_model.recommend(
        unknown_user, k=5, context_items=items, context_ratings=[5] * 4
    )
    assert folded.strategy == "fold_in"
    assert not set(items) & set(folded.items)
    known = int(prepared.split.train_val["user_id"].iloc[0])
    realtime = implicit_model.recommend(known, k=5, context_items=items, context_ratings=[5] * 4)
    assert realtime.strategy == "personalized_realtime"
    # items the model has never seen are ignored, not an error
    assert implicit_model.recommend(unknown_user, k=5, context_items=[-1]).strategy == "popularity"


def test_fold_in_reproduces_the_user_embedding(prepared) -> None:
    model = ImplicitALS(factors=8, regularization=1.0, alpha=5.0, iterations=15, solver="exact")
    model.fit(prepared.split.train_val)
    uidx = 0
    hist_idx, hist_r = model._history(uidx)
    folded = model.fold_in(hist_idx, hist_r)
    stored = model.user_factors[uidx]
    cosine = folded @ stored / (np.linalg.norm(folded) * np.linalg.norm(stored))
    assert cosine > 0.95


def test_blend_weight_mixes_in_popularity(prepared) -> None:
    pure = ImplicitALS(factors=8, iterations=4, blend_weight=0.0).fit(prepared.split.train_val)
    blended = ImplicitALS(factors=8, iterations=4, blend_weight=100.0).fit(prepared.split.train_val)
    user = int(prepared.split.train_val["user_id"].iloc[0])
    popular = set(blended.recommend_popular(30).items)
    assert len(popular & set(blended.recommend(user, 10).items)) >= len(
        popular & set(pure.recommend(user, 10).items)
    )


def test_batch_and_single_recommendations_agree(implicit_model) -> None:
    uidx = np.arange(5)
    batch = implicit_model.recommend_batch(uidx, 10)
    for row, user_index in zip(batch, uidx, strict=True):
        user_id = int(implicit_model.user_index.ids[user_index])
        single = implicit_model.recommend(user_id, 10).items
        assert implicit_model.item_index.ids[row].tolist() == single


@pytest.mark.parametrize("model_type", ["popularity", "explicit_als", "implicit_als"])
def test_save_load_roundtrip(tmp_path, prepared, model_type) -> None:
    model = build_model(ModelConfig(type=model_type, factors=4, iterations=2)).fit(
        prepared.split.train
    )
    model.save(tmp_path / "m")
    loaded = load_model(tmp_path / "m")
    user = int(prepared.split.train["user_id"].iloc[0])
    assert loaded.recommend(user, 10).items == model.recommend(user, 10).items
    assert loaded.metadata["n_ratings"] == model.metadata["n_ratings"]


def test_csr_sums_duplicates() -> None:
    m = _csr(np.array([0, 0]), np.array([1, 1]), np.array([2.0, 3.0]), (1, 2))
    assert m[0, 1] == 5.0


@pytest.mark.parametrize("model_cls", ["implicit", "explicit"])
def test_torch_backend_matches_numpy(prepared, model_cls) -> None:
    pytest.importorskip("torch")
    from recsys.models import ExplicitALS, ImplicitALS

    train = prepared.split.train_val
    if model_cls == "implicit":
        params = {"factors": 8, "regularization": 1.0, "alpha": 5.0, "iterations": 4}
        numpy_model = ImplicitALS(**params).fit(train)
        torch_model = ImplicitALS(**params, backend="torch", device="cpu").fit(train)
    else:
        params = {"factors": 8, "regularization": 0.05, "iterations": 4}
        numpy_model = ExplicitALS(**params).fit(train)
        torch_model = ExplicitALS(**params, backend="torch", device="cpu").fit(train)
    users = np.arange(numpy_model.n_users)
    # float32 on the torch side vs float64 in numpy: equal up to rounding (scores are O(1);
    # differences stay below 2e-3 even after the CG steps), and the same rankings.
    np.testing.assert_allclose(
        torch_model.score_users(users), numpy_model.score_users(users), rtol=1e-2, atol=2e-3
    )
    same = np.mean(
        [
            torch_model.recommend(int(u), k=5).items == numpy_model.recommend(int(u), k=5).items
            for u in numpy_model.user_index.ids[:50]
        ]
    )
    assert same >= 0.95
