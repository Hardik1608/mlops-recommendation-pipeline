"""SASRec: sequence handling, causality, ONNX export, persistence, real-time context.

Needs torch (``requirements-gpu.txt``, or the CPU wheel); skipped when it is missing. Runs
on CPU in a few seconds on a planted "chain" dataset: every item is followed by the next
one (mod N), which a sequential model must learn and an order-blind model cannot.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnxruntime")

from recsys.models import ImplicitALS, SASRec  # noqa: E402
from recsys.models.io import load_model  # noqa: E402
from recsys.models.sasrec_torch import SASRecNet  # noqa: E402

N_ITEMS = 30
ITEM_BASE = 1000  # raw ids 1000..1029


def chain_ratings(n_users: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for user in range(n_users):
        start, length = int(rng.integers(N_ITEMS)), int(rng.integers(8, 20))
        for t in range(length):
            item = ITEM_BASE + (start + t) % N_ITEMS
            rows.append((user, item, 4.0, 1_000_000 + user * 100 + t))
    return pd.DataFrame(rows, columns=["user_id", "item_id", "rating", "timestamp"])


def successor(raw_item: int, steps: int = 1) -> int:
    return ITEM_BASE + (raw_item - ITEM_BASE + steps) % N_ITEMS


SMALL = {
    "factors": 32,
    "max_len": 20,
    "num_blocks": 2,
    "num_heads": 2,
    "dropout": 0.1,
    "learning_rate": 5e-3,
    "batch_size": 64,
    "epochs": 40,
    "device": "cpu",
    "seed": 7,
}


@pytest.fixture(scope="module")
def chain() -> pd.DataFrame:
    return chain_ratings()


@pytest.fixture(scope="module")
def sasrec(chain: pd.DataFrame) -> SASRec:
    return SASRec(**SMALL).fit(chain)


def last_items(ratings: pd.DataFrame) -> dict[int, int]:
    ordered = ratings.sort_values("timestamp")
    return {int(u): int(g["item_id"].iloc[-1]) for u, g in ordered.groupby("user_id")}


def test_learns_the_next_item_that_order_blind_models_cannot(chain, sasrec) -> None:
    last = last_items(chain)
    users = sorted(last)[:100]
    sas_hits = np.mean([sasrec.recommend(u, k=1).items[0] == successor(last[u]) for u in users])
    als = ImplicitALS(factors=16, regularization=1.0, iterations=10).fit(chain)
    als_hits = np.mean([als.recommend(u, k=1).items[0] == successor(last[u]) for u in users])
    assert sas_hits >= 0.9
    assert sas_hits > als_hits + 0.3  # ALS sees the set of items, not their order


def test_user_sequences_are_time_ordered_and_left_padded(chain, sasrec) -> None:
    user = 0
    history = chain[chain["user_id"] == user].sort_values("timestamp")["item_id"].to_numpy()
    row = sasrec.user_seq[sasrec.user_index.get(user)]
    expected = sasrec.item_index.lookup(history[-SMALL["max_len"] :]) + 1
    assert (row[: SMALL["max_len"] - len(expected)] == 0).all()
    np.testing.assert_array_equal(row[SMALL["max_len"] - len(expected) :], expected)


def test_streamed_events_extend_the_sequence(chain, sasrec) -> None:
    user = 3
    last = last_items(chain)[user]
    context = [successor(last), successor(last, 2)]
    rec = sasrec.recommend(user, k=1, context_items=context)
    assert rec.strategy == "personalized_realtime"
    assert rec.items[0] == successor(last, 3)


def test_new_user_is_scored_from_streamed_events_alone(sasrec) -> None:
    rec = sasrec.recommend(99_999, k=1, context_items=[1005, 1006, 1007])
    assert rec.strategy == "fold_in"
    assert rec.items[0] == 1008


def test_onnx_export_matches_pytorch(sasrec) -> None:
    assert sasrec.metadata["onnx_max_abs_diff"] < 1e-3
    assert sasrec.onnx_model.size > 0


def test_save_load_roundtrip_serves_the_same_recommendations(tmp_path, sasrec) -> None:
    loaded = load_model(sasrec.save(tmp_path / "sasrec"))
    assert isinstance(loaded, SASRec)
    for user in (0, 5, 42):
        a, b = sasrec.recommend(user, k=5), loaded.recommend(user, k=5)
        assert a.items == b.items
        np.testing.assert_allclose(a.scores, b.scores, rtol=1e-5, atol=1e-5)


def test_batch_and_single_scoring_agree(sasrec) -> None:
    users = np.arange(8)
    top = sasrec.recommend_batch(users, 3)
    for row, uidx in zip(top, users, strict=True):
        single = sasrec.recommend(int(sasrec.user_index.ids[uidx]), k=3)
        assert sasrec.item_index.ids[row].tolist() == single.items


def test_attention_is_causal_and_padding_is_safe() -> None:
    torch.manual_seed(0)
    net = SASRecNet(n_items=50, hidden=16, max_len=10, blocks=2, heads=2, dropout=0.0).eval()
    seq = torch.tensor([[0, 0, 0, 0, 5, 9, 3, 7, 1, 2]])
    changed = seq.clone()
    changed[0, -1] = 40  # only the newest item differs
    with torch.no_grad():
        a, b = net.encode(seq), net.encode(changed)
        lone = net.encode(torch.tensor([[0] * 9 + [4]]))
    torch.testing.assert_close(a[:, :-1], b[:, :-1])  # earlier positions cannot see it
    assert not torch.equal(a[:, -1], b[:, -1])
    assert torch.isfinite(lone).all()  # a fully padded prefix produces no NaNs


def test_early_stopping_keeps_the_best_epoch(chain) -> None:
    values = iter([0.1, 0.3, 0.2, 0.25, 0.1, 0.1])
    model = SASRec(**{**SMALL, "epochs": 10, "eval_every": 1, "patience": 2})
    model.set_monitor(lambda m: next(values))
    model.fit(chain)
    assert model.metadata["best_epoch"] == 2
    assert model.metadata["epochs_run"] == 4  # stopped after two checks without gain
    assert [h.get("val_metric") for h in model.history] == [0.1, 0.3, 0.2, 0.25]


def test_monitor_scores_with_the_live_network(chain) -> None:
    seen: list[tuple[int, int]] = []

    def monitor(model: SASRec) -> float:
        seen.append(model.score_users(np.arange(3)).shape)
        return 0.0

    model = SASRec(**{**SMALL, "epochs": 2, "eval_every": 1, "patience": 5})
    model.set_monitor(monitor)
    model.fit(chain)
    assert seen == [(3, N_ITEMS), (3, N_ITEMS)]
    assert model._scorer is None  # cleared again: serving uses the ONNX export


def test_shuffled_order_ablation_loses_the_sequence_signal(chain, sasrec) -> None:
    shuffled = SASRec(**{**SMALL, "sequence_order": "shuffled"}).fit(chain)
    last = last_items(chain)
    users = sorted(last)[:100]

    def hits(model: SASRec) -> float:
        return float(
            np.mean([model.recommend(u, k=1).items[0] == successor(last[u]) for u in users])
        )

    assert hits(sasrec) > hits(shuffled) + 0.3
    assert shuffled.metadata["onnx_max_abs_diff"] < 1e-3


def test_chunked_cross_entropy_equals_the_full_softmax() -> None:
    from recsys.models import sasrec_torch

    torch.manual_seed(0)
    n = sasrec_torch.CE_CHUNK * 2 + 17  # several slices plus a ragged one
    h = torch.randn(n, 16, requires_grad=True)
    w = torch.randn(50, 16, requires_grad=True)
    target = torch.randint(0, 50, (n,))
    expected = torch.nn.functional.cross_entropy(h @ w.T, target)
    expected.backward()
    grads = h.grad.clone(), w.grad.clone()
    h.grad = w.grad = None
    got = sasrec_torch.full_softmax_ce(h, w, target)
    got.backward()
    torch.testing.assert_close(got, expected)
    torch.testing.assert_close(h.grad, grads[0])
    torch.testing.assert_close(w.grad, grads[1])


def test_sampled_softmax_trains(chain) -> None:
    model = SASRec(**{**SMALL, "loss": "sampled_ce", "num_negatives": 8, "epochs": 3})
    model.fit(chain)
    losses = [h["loss"] for h in model.history]
    assert all(np.isfinite(losses))


def test_leave_one_out_check_runs_early_stopping_then_refits(chain) -> None:
    from recsys.config import ModelConfig
    from recsys.evaluation.protocols import leave_one_out_check

    sasrec_cfg = ModelConfig(type="sasrec", **{**SMALL, "eval_every": 5, "patience": 3})
    models = {
        "implicit_als": ModelConfig(type="implicit_als", factors=16, regularization=1.0),
        "sasrec": sasrec_cfg,
    }
    out = leave_one_out_check(chain, models, k=10)
    assert out["sasrec"]["n_users"] == chain["user_id"].nunique()
    assert out["sasrec"]["ndcg"] > 0.95  # the next item of a chain is fully predictable
    # ALS ranks both unseen neighbours of the user's chain segment high but cannot tell
    # which one comes next, so it scores lower without being useless.
    assert out["sasrec"]["ndcg"] > out["implicit_als"]["ndcg"]


def test_serving_imports_do_not_need_torch() -> None:
    code = (
        "import sys; import recsys.models, recsys.models.io, recsys.serving.service; "
        "assert 'torch' not in sys.modules, 'torch imported'"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_interrupted_training_resumes_to_the_same_result(chain, tmp_path, monkeypatch) -> None:
    from recsys.models import sasrec_torch

    params = {**SMALL, "epochs": 6, "eval_every": 2, "patience": 9}
    reference = SASRec(**params)
    reference.set_monitor(lambda m: float(m.score_users(np.arange(5)).mean()))
    reference.fit(chain)

    real_loss, calls = sasrec_torch._batch_loss, {"n": 0}

    def crash_in_epoch_four(*args, **kwargs):  # 5 batches per epoch on this data
        calls["n"] += 1
        if calls["n"] == 3 * 5 + 2:
            raise RuntimeError("power cut")
        return real_loss(*args, **kwargs)

    path = tmp_path / "ckpt.pt"
    interrupted = SASRec(**params)
    interrupted.set_monitor(lambda m: float(m.score_users(np.arange(5)).mean()))
    interrupted.set_checkpoint(path)
    monkeypatch.setattr(sasrec_torch, "_batch_loss", crash_in_epoch_four)
    with pytest.raises(RuntimeError, match="power cut"):
        interrupted.fit(chain)
    assert path.exists()  # epochs 1-3 are safe on disk
    monkeypatch.setattr(sasrec_torch, "_batch_loss", real_loss)

    resumed = SASRec(**params)
    resumed.set_monitor(lambda m: float(m.score_users(np.arange(5)).mean()))
    resumed.set_checkpoint(path)
    resumed.fit(chain)
    assert not path.exists()  # removed once the run finished
    assert [h["loss"] for h in resumed.history] == pytest.approx(
        [h["loss"] for h in reference.history], rel=1e-6
    )
    assert resumed.metadata["best_epoch"] == reference.metadata["best_epoch"]
    np.testing.assert_allclose(
        resumed.score_users(np.arange(8)), reference.score_users(np.arange(8)), atol=1e-5
    )


def test_checkpoint_for_other_hyper_parameters_is_ignored(chain, tmp_path) -> None:
    path = tmp_path / "ckpt.pt"
    first = SASRec(**{**SMALL, "epochs": 2})
    first.set_checkpoint(path)
    first.fit(chain)  # finishes and removes its checkpoint
    from recsys.models import sasrec_torch

    torch.save({"version": sasrec_torch.CHECKPOINT_VERSION, "signature": {"other": 1}}, path)
    second = SASRec(**{**SMALL, "epochs": 2})
    second.set_checkpoint(path)
    second.fit(chain)  # trains from scratch instead of loading a mismatched state
    assert len(second.history) == 2
