"""SASRec: self-attentive sequential recommendation (Kang & McAuley, 2018).

Training runs in PyTorch (:mod:`recsys.models.sasrec_torch`, on a GPU when one is
available). The trained network is exported to ONNX and stored in the model package, so
scoring only needs ``onnxruntime``: the API image carries no torch.

Unlike matrix factorisation, the model reads a user's interactions *in time order*; the user
representation is the network state after the most recent ``max_len`` items. Fresh
(streamed) events are appended to that sequence, so real-time personalisation is the model's
native input rather than a fold-in approximation, and a brand-new user with a few events is
scored the same way.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from recsys.models.base import Recommender

Scorer = Callable[[np.ndarray], np.ndarray]
# ONNX Runtime's CPU arena grows to the largest batch it has run and keeps that memory for
# the session's lifetime: 2.4 GB at 512 sequences (hidden 128, max_len 200), 0.6 GB at 128,
# for a 7% slower scoring pass (measured). Serving runs batch 1, so its arena stays small.
SCORE_BATCH = 128


class SASRec(Recommender):
    model_type = "sasrec"

    def __init__(self, **params: Any) -> None:
        params.setdefault("factors", 64)  # hidden size
        params.setdefault("max_len", 200)
        params.setdefault("num_blocks", 2)
        params.setdefault("num_heads", 1)
        params.setdefault("dropout", 0.2)
        params.setdefault("learning_rate", 1e-3)
        params.setdefault("batch_size", 128)
        params.setdefault("epochs", 200)
        params.setdefault("loss", "ce")
        params.setdefault("num_negatives", 1024)
        params.setdefault("eval_every", 5)
        params.setdefault("patience", 5)
        params.setdefault("device", "auto")
        params.setdefault("seed", 42)
        params.setdefault("sequence_order", "time")  # "shuffled": order ablation
        super().__init__(**params)
        # Most recent training items per user, left-padded; ids are item index + 1 (0 = pad).
        self.user_seq = np.zeros((0, self.max_len), dtype=np.int32)
        self.onnx_model = np.zeros(0, dtype=np.uint8)
        self.history: list[dict[str, float]] = []
        self.monitor: Callable[[Recommender], float] | None = None
        self.checkpoint_path: Path | None = None  # set by long-running callers
        self._scorer: Scorer | None = None  # live torch network, only while training
        self._session: Any = None

    @property
    def max_len(self) -> int:
        return int(self.params["max_len"])

    def set_monitor(self, monitor: Callable[[Recommender], float]) -> None:
        """Validation metric (higher is better) for early stopping during ``fit``."""
        self.monitor = monitor

    def set_checkpoint(self, path: str | Path) -> None:
        """Write a resumable training checkpoint here after every epoch."""
        self.checkpoint_path = Path(path)

    # ------------------------------------------------------------------ training
    def _fit(self, u: np.ndarray, i: np.ndarray, r: np.ndarray, ts: np.ndarray) -> None:
        from recsys.models import sasrec_torch  # torch is a training-only dependency

        inputs, targets = self._build_sequences(u, i)
        monitor = None
        if self.monitor is not None:
            evaluate = self.monitor

            def monitor(scorer: Scorer) -> float:
                self._scorer = scorer
                try:
                    return float(evaluate(self))
                finally:
                    self._scorer = None

        out = sasrec_torch.train(
            inputs, targets, self.n_items, self.params, monitor, self.checkpoint_path
        )
        self.onnx_model = np.frombuffer(out.onnx_model, dtype=np.uint8).copy()
        self._session = None
        self.history = out.history
        self.metadata.update(
            best_epoch=out.best_epoch,
            epochs_run=len(out.history),
            device=out.device,
            peak_vram_gb=out.peak_vram_gb,
            onnx_max_abs_diff=out.onnx_max_abs_diff,
            onnx_mb=round(len(out.onnx_model) / 1e6, 2),
            train_sequences=len(inputs),
        )

    def _build_sequences(self, u: np.ndarray, i: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Per-user time-ordered sequences -> ``user_seq`` and (input, target) training pairs.

        ``fit`` passes interactions sorted by time, so a stable sort by user keeps each
        user's events in order. Training uses the last ``max_len + 1`` events of every user
        with at least two: inputs are events ``t``, targets events ``t + 1``. With
        ``sequence_order="shuffled"`` each user's events are randomly permuted first.
        """
        length = self.max_len
        rng = np.random.default_rng(int(self.params["seed"]))
        shuffle = self.params.get("sequence_order") == "shuffled"
        order = np.argsort(u, kind="stable")
        users = u[order]
        items = i[order].astype(np.int64) + 1  # 0 is padding
        bounds = np.searchsorted(users, np.arange(self.n_users + 1))
        counts = np.diff(bounds)
        self.user_seq = np.zeros((self.n_users, length), dtype=np.int32)
        inputs = np.zeros((int((counts >= 2).sum()), length), dtype=np.int32)
        targets = np.zeros_like(inputs)
        row = 0
        for user in range(self.n_users):
            seq = items[bounds[user] : bounds[user + 1]]
            if shuffle:
                seq = rng.permutation(seq)
            tail = seq[-length:]
            self.user_seq[user, length - len(tail) :] = tail
            if len(seq) >= 2:
                window = seq[-(length + 1) :]
                n = len(window) - 1
                inputs[row, length - n :] = window[:-1]
                targets[row, length - n :] = window[1:]
                row += 1
        return inputs, targets

    # ------------------------------------------------------------------ scoring
    def _ort_session(self) -> Any:
        if self._session is None:
            if self.onnx_model.size == 0:
                raise RuntimeError("SASRec has no trained network (fit or load it first)")
            import onnxruntime as ort

            options = ort.SessionOptions()
            threads = int(os.environ.get("RECSYS_ORT_THREADS", "0"))
            if threads > 0:
                options.intra_op_num_threads = threads
            self._session = ort.InferenceSession(
                self.onnx_model.tobytes(), options, providers=["CPUExecutionProvider"]
            )
        return self._session

    def _score(self, seqs: np.ndarray) -> np.ndarray:
        if self._scorer is not None:
            return self._scorer(seqs)
        session = self._ort_session()
        out = [
            session.run(["scores"], {"seq": seqs[start : start + SCORE_BATCH].astype(np.int64)})[0]
            for start in range(0, len(seqs), SCORE_BATCH)
        ]
        return np.concatenate(out) if out else np.zeros((0, self.n_items), dtype=np.float32)

    def score_users(self, user_idx: np.ndarray) -> np.ndarray:
        return self._score(self.user_seq[np.asarray(user_idx)])

    def score_context(self, item_idx: np.ndarray, ratings: np.ndarray) -> np.ndarray:
        """Score the sequence ``item_idx`` (oldest first); ratings are not used."""
        seq = np.zeros((1, self.max_len), dtype=np.int64)
        tail = np.asarray(item_idx, dtype=np.int64)[-self.max_len :] + 1
        if len(tail):
            seq[0, self.max_len - len(tail) :] = tail
        return self._score(seq)[0]

    def _context_history(self, uidx: int) -> tuple[np.ndarray, np.ndarray]:
        row = self.user_seq[uidx]
        items = row[row > 0].astype(np.int64) - 1
        return items, np.ones(len(items))

    # ------------------------------------------------------------------ persistence
    def _arrays(self) -> dict[str, np.ndarray]:
        return {"user_seq": self.user_seq, "onnx_model": self.onnx_model}

    def _load_arrays(self, arrays: dict[str, np.ndarray]) -> None:
        self.user_seq = arrays["user_seq"].astype(np.int32)
        self.onnx_model = arrays["onnx_model"].astype(np.uint8)
        self._session = None
