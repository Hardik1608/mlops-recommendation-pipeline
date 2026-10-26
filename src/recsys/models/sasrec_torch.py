"""SASRec training in PyTorch, exported to ONNX for CPU serving.

SASRec (Kang & McAuley, 2018) is a causal self-attention model over a user's time-ordered
interactions: the hidden state at position t predicts the item at t + 1. Two losses:

* ``ce``         - full softmax cross-entropy over the catalogue. Klenitskiy & Vasilev (2023)
                   show it is much stronger than the original one-negative BCE loss.
* ``sampled_ce`` - softmax over the positive and ``num_negatives`` uniformly sampled items
                   shared by the batch, for catalogues where full logits do not fit in memory.

Imported only by :meth:`recsys.models.sasrec.SASRec._fit`; torch is a training-time
dependency (``requirements-gpu.txt``) and never needed by the API.
"""

from __future__ import annotations

import copy
import io
import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812 (PyTorch convention)
from torch import nn
from torch.utils.checkpoint import checkpoint as recompute_in_backward

logger = logging.getLogger(__name__)

MASK = -1e9  # additive attention mask; finite so bf16/fp16 softmax never sees inf - inf
Scorer = Callable[[np.ndarray], np.ndarray]


class CausalSelfAttention(nn.Module):
    """Plain multi-head attention (explicit ops, so the ONNX export is a simple graph)."""

    def __init__(self, hidden: int, heads: int, dropout: float) -> None:
        super().__init__()
        if hidden % heads:
            raise ValueError(f"hidden size {hidden} is not divisible by {heads} heads")
        self.heads, self.head_dim = heads, hidden // heads
        self.qkv = nn.Linear(hidden, 3 * hidden)
        self.out = nn.Linear(hidden, hidden)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        b, length, hidden = x.shape
        qkv = self.qkv(x).view(b, length, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (b, heads, L, d)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim) + bias
        att = self.drop(att.softmax(dim=-1))
        return self.out((att @ v).transpose(1, 2).reshape(b, length, hidden))


class Block(nn.Module):
    """Pre-norm transformer block; padded positions are zeroed after every block."""

    def __init__(self, hidden: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden)
        self.attn = CausalSelfAttention(hidden, heads, dropout)
        self.ln2 = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, hidden)
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, bias: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.attn(self.ln1(x), bias))
        x = x + self.drop(self.ffn(self.ln2(x)))
        return x * keep


class SASRecNet(nn.Module):
    """Item ids are ``1..n_items``; 0 is padding. Sequences are left-padded to ``max_len``,
    so the most recent interaction is always at the last position."""

    def __init__(
        self, n_items: int, hidden: int, max_len: int, blocks: int, heads: int, dropout: float
    ) -> None:
        super().__init__()
        self.item_emb = nn.Embedding(n_items + 1, hidden, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, hidden)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(Block(hidden, heads, dropout) for _ in range(blocks))
        self.norm = nn.LayerNorm(hidden)
        self.scale = math.sqrt(hidden)
        causal = torch.full((max_len, max_len), MASK).triu(1)
        self.register_buffer("causal", causal, persistent=False)
        self.register_buffer("eye", torch.eye(max_len), persistent=False)
        nn.init.normal_(self.item_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()

    def item_weights(self) -> torch.Tensor:
        return self.item_emb.weight[1:]

    def encode(self, seq: torch.Tensor) -> torch.Tensor:
        """(B, L) item ids -> (B, L, hidden) states; position t has seen items <= t."""
        length = seq.shape[1]
        pos = torch.arange(length, device=seq.device)
        x = self.drop(self.item_emb(seq) * self.scale + self.pos_emb(pos))
        keep = (seq > 0).unsqueeze(-1).to(x.dtype)
        x = x * keep
        # Causal mask + padded keys. The diagonal stays open, so a padded query attends to
        # itself (its output is zeroed anyway) instead of a fully masked row (NaN).
        pad_keys = (seq == 0).to(x.dtype)[:, None, None, :] * MASK
        bias = self.causal[:length, :length] + pad_keys * (1.0 - self.eye[:length, :length])
        for block in self.blocks:
            x = block(x, bias, keep)
        return self.norm(x)


class LastPositionScorer(nn.Module):
    """ONNX entry point: ``seq`` (B, L) int64 -> ``scores`` (B, n_items) float32."""

    def __init__(self, net: SASRecNet) -> None:
        super().__init__()
        self.net = net

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        return self.net.encode(seq)[:, -1, :] @ self.net.item_weights().T


@dataclass
class TrainOutput:
    onnx_model: bytes
    history: list[dict[str, float]]
    best_epoch: int
    device: str
    peak_vram_gb: float | None
    onnx_max_abs_diff: float
    extra: dict[str, Any] = field(default_factory=dict)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


CE_CHUNK = 2048  # positions per cross-entropy slice


def _chunk_ce_sum(h: torch.Tensor, weights: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy((h @ weights.T).float(), target, reduction="sum")


def full_softmax_ce(h: torch.Tensor, weights: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean cross-entropy over all items, computed in slices of ``CE_CHUNK`` positions.

    Each slice's (positions x items) logits are freed after its forward pass and recomputed
    in the backward pass (activation checkpointing). Peak memory is one slice instead of the
    whole batch: about 2 GB instead of ~12 GB for 12,800 positions x 59k items (ml-25m).
    """
    if len(h) <= CE_CHUNK:
        return _chunk_ce_sum(h, weights, target) / len(h)
    total = h.new_zeros((), dtype=torch.float32)
    for start in range(0, len(h), CE_CHUNK):
        part = slice(start, start + CE_CHUNK)
        total = total + recompute_in_backward(
            _chunk_ce_sum, h[part], weights, target[part], use_reentrant=False
        )
    return total / len(h)


def _batch_loss(
    net: SASRecNet, inp: torch.Tensor, tgt: torch.Tensor, loss: str, num_negatives: int
) -> torch.Tensor:
    hidden = net.encode(inp)
    valid = tgt > 0
    h = hidden[valid]  # (N, hidden)
    target = tgt[valid] - 1  # 0-based item index
    weights = net.item_weights()
    if loss == "ce":
        return full_softmax_ce(h, weights, target)
    negatives = torch.randint(0, weights.shape[0], (num_negatives,), device=h.device)
    pos = (h * weights[target]).sum(dim=-1, keepdim=True)
    neg = h @ weights[negatives].T
    neg = neg.masked_fill(negatives[None, :] == target[:, None], MASK)  # accidental hits
    logits = torch.cat([pos, neg], dim=1).float()
    return F.cross_entropy(logits, torch.zeros(len(target), dtype=torch.long, device=h.device))


def make_scorer(net: SASRecNet, device: torch.device, batch_size: int = 512) -> Scorer:
    """Score left-padded sequences with the live network (used for validation checks)."""
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()

    def score(seqs: np.ndarray) -> np.ndarray:
        was_training = net.training
        net.eval()
        out = []
        with torch.no_grad(), torch.autocast(device.type, torch.bfloat16, enabled=use_bf16):
            for start in range(0, len(seqs), batch_size):
                seq = torch.from_numpy(seqs[start : start + batch_size]).long().to(device)
                h = net.encode(seq)[:, -1, :]
                out.append((h @ net.item_weights().T).float().cpu().numpy())
        net.train(was_training)
        return np.concatenate(out) if out else np.zeros((0, net.item_weights().shape[0]))

    return score


def export_onnx(net: SASRecNet, max_len: int) -> bytes:
    """Export the last-position scorer with a dynamic batch dimension.

    Uses the ``torch.export``-based exporter (needs ``onnxscript``); falls back to the
    deprecated TorchScript exporter where that is unavailable. Either way the result is
    checked against PyTorch by :func:`check_onnx`.
    """
    scorer = LastPositionScorer(copy.deepcopy(net).float().cpu()).eval()
    example = torch.ones((2, max_len), dtype=torch.long)
    names: dict[str, Any] = {"input_names": ["seq"], "output_names": ["scores"]}
    with torch.no_grad():
        try:
            batch = torch.export.Dim("batch", min=1, max=65_536)
            program = torch.onnx.export(
                scorer,
                (example,),
                dynamo=True,
                dynamic_shapes={"seq": {0: batch}},
                opset_version=18,
                verbose=False,
                **names,
            )
            return program.model_proto.SerializeToString()
        except Exception as exc:  # missing onnxscript, or an exporter limitation
            logger.warning("torch.export-based ONNX export failed (%s); using TorchScript", exc)
        buffer = io.BytesIO()
        torch.onnx.export(
            scorer,
            (example,),
            buffer,
            dynamo=False,
            dynamic_axes={"seq": {0: "batch"}, "scores": {0: "batch"}},
            opset_version=17,
            **names,
        )
    return buffer.getvalue()


def check_onnx(onnx_model: bytes, net: SASRecNet, seqs: np.ndarray) -> float:
    """Max |ONNX - PyTorch| score difference on real sequences (both float32 on CPU)."""
    import onnxruntime as ort

    session = ort.InferenceSession(onnx_model, providers=["CPUExecutionProvider"])
    ref = LastPositionScorer(copy.deepcopy(net).float().cpu().eval())
    with torch.no_grad():
        expected = ref(torch.from_numpy(seqs).long()).numpy()
    got = session.run(["scores"], {"seq": seqs.astype(np.int64)})[0]
    return float(np.abs(got - expected).max())


CHECKPOINT_VERSION = 1


def _signature(params: dict[str, Any], n_items: int, n_sequences: int) -> dict[str, Any]:
    """What a checkpoint must match to be resumed: the data shape and every hyper-parameter
    except the epoch budget (a run can be resumed with more epochs) and the device."""
    kept = {k: params[k] for k in sorted(params) if k not in ("epochs", "device")}
    return {"params": kept, "n_items": n_items, "n_sequences": n_sequences}


def _save_checkpoint(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)  # atomic rename: a crash mid-write never leaves a half-written file


def train(
    inputs: np.ndarray,
    targets: np.ndarray,
    n_items: int,
    params: dict[str, Any],
    monitor: Callable[[Scorer], float] | None = None,
    checkpoint: str | Path | None = None,
) -> TrainOutput:
    """Fit SASRec on (input, target) sequence pairs and return the ONNX model.

    With ``monitor`` (validation NDCG, higher is better), the network is checked every
    ``eval_every`` epochs, training stops after ``patience`` checks without improvement and
    the best weights are kept. Without it, training runs for exactly ``epochs`` epochs
    (the count chosen on the validation window).

    With ``checkpoint``, the full training state (weights, optimiser, random-number
    generators, history, early-stopping state) is written after every epoch and a matching
    checkpoint is resumed, so an interrupted run (power cut, OOM kill) loses at most one
    epoch and ends with the same result as an uninterrupted one. It is deleted on success.
    """
    seed = int(params["seed"])
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = resolve_device(str(params["device"]))
    max_len = int(params["max_len"])
    net = SASRecNet(
        n_items,
        int(params["factors"]),
        max_len,
        int(params["num_blocks"]),
        int(params["num_heads"]),
        float(params["dropout"]),
    ).to(device)
    optimiser = torch.optim.Adam(
        net.parameters(), lr=float(params["learning_rate"]), betas=(0.9, 0.98)
    )
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    inputs_t = torch.from_numpy(inputs.astype(np.int64))
    targets_t = torch.from_numpy(targets.astype(np.int64))
    batch_size, loss_type = int(params["batch_size"]), str(params["loss"])
    num_negatives = int(params["num_negatives"])
    epochs, eval_every = int(params["epochs"]), int(params["eval_every"])
    patience = int(params["patience"])

    history: list[dict[str, float]] = []
    best_value, best_epoch, best_state, checks_without_gain = -math.inf, epochs, None, 0
    first_epoch = 1
    signature = _signature(params, n_items, len(inputs))
    checkpoint = Path(checkpoint) if checkpoint is not None else None
    if checkpoint is not None and checkpoint.exists():
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        if state.get("version") == CHECKPOINT_VERSION and state.get("signature") == signature:
            net.load_state_dict(state["net"])
            optimiser.load_state_dict(state["optimiser"])
            rng.bit_generator.state = state["numpy_rng"]
            torch.set_rng_state(state["torch_rng"].cpu())
            if device.type == "cuda" and state.get("cuda_rng") is not None:
                torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda_rng"]])
            history = state["history"]
            best_value, best_epoch = state["best_value"], state["best_epoch"]
            best_state, checks_without_gain = state["best_state"], state["checks_without_gain"]
            first_epoch = state["epoch"] + 1
            if monitor is not None and checks_without_gain >= patience:
                first_epoch = epochs + 1  # the run had already stopped early
            logger.info("resumed from %s after epoch %d", checkpoint, state["epoch"])
        else:
            logger.warning("ignoring checkpoint %s: different data or hyper-parameters", checkpoint)
    for epoch in range(first_epoch, epochs + 1):
        start = time.perf_counter()
        net.train()
        order = torch.from_numpy(rng.permutation(len(inputs_t)))
        total, batches = 0.0, 0
        for begin in range(0, len(order), batch_size):
            idx = order[begin : begin + batch_size]
            inp = inputs_t[idx].to(device, non_blocking=True)
            tgt = targets_t[idx].to(device, non_blocking=True)
            with torch.autocast(device.type, torch.bfloat16, enabled=use_bf16):
                loss = _batch_loss(net, inp, tgt, loss_type, num_negatives)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            total += float(loss.detach())
            batches += 1
        step: dict[str, float] = {
            "iteration": epoch,
            "seconds": time.perf_counter() - start,
            "loss": total / max(batches, 1),
        }
        if monitor is not None and (epoch % eval_every == 0 or epoch == epochs):
            value = monitor(make_scorer(net, device))
            step["val_metric"] = value
            if value > best_value:
                best_value, best_epoch, checks_without_gain = value, epoch, 0
                best_state = copy.deepcopy(net.state_dict())
            else:
                checks_without_gain += 1
            logger.info(
                "epoch %d loss=%.4f val=%.4f (best %.4f @ %d)",
                epoch,
                step["loss"],
                value,
                best_value,
                best_epoch,
            )
        else:
            logger.info("epoch %d loss=%.4f (%.1fs)", epoch, step["loss"], step["seconds"])
        history.append(step)
        if checkpoint is not None:
            _save_checkpoint(
                checkpoint,
                {
                    "version": CHECKPOINT_VERSION,
                    "signature": signature,
                    "epoch": epoch,
                    "net": net.state_dict(),
                    "optimiser": optimiser.state_dict(),
                    "numpy_rng": rng.bit_generator.state,
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                    "history": history,
                    "best_value": best_value,
                    "best_epoch": best_epoch,
                    "best_state": best_state,
                    "checks_without_gain": checks_without_gain,
                },
            )
        if monitor is not None and checks_without_gain >= patience:
            logger.info("early stop at epoch %d; best epoch %d", epoch, best_epoch)
            break

    if best_state is not None:
        net.load_state_dict(best_state)
    if monitor is None:
        best_epoch = len(history)
    peak = None
    if device.type == "cuda":
        peak = round(torch.cuda.max_memory_allocated(device) / 1024**3, 3)

    onnx_model = export_onnx(net, max_len)
    sample = inputs[rng.choice(len(inputs), size=min(64, len(inputs)), replace=False)]
    diff = check_onnx(onnx_model, net, sample)
    logger.info("ONNX export: %.1f MB, max |onnx - torch| = %.2e", len(onnx_model) / 1e6, diff)
    if checkpoint is not None:
        checkpoint.unlink(missing_ok=True)
    return TrainOutput(
        onnx_model=onnx_model,
        history=history,
        best_epoch=best_epoch,
        device=str(device),
        peak_vram_gb=peak,
        onnx_max_abs_diff=diff,
    )
