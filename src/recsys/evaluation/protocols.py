"""Leave-one-out evaluation: the protocol most sequential-recommendation papers report.

Each user's last interaction is the test item, the second-to-last is for validation, the
rest is training data. Every interaction counts as relevant. Users are cut at *their own*
last event, so training data includes events that happened after other users' test events
(Ji et al. 2023, "A Critical Study on Data Leakage in Recommender System Offline
Evaluation"). The benchmark runs it as a protocol check next to the global temporal split
that this project reports, to show how much the choice of protocol moves the conclusions.
"""

from __future__ import annotations

import gc
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from recsys.config import ModelConfig
from recsys.evaluation.evaluate import evaluate_ranking
from recsys.models.io import build_model

logger = logging.getLogger(__name__)

ANY_INTERACTION = 0.0  # relevance threshold: every held-out event is relevant


def leave_one_out_split(
    ratings: pd.DataFrame, min_events: int = 3
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(train, val, test). Users with fewer than ``min_events`` events stay in training."""
    ordered = ratings.sort_values(["user_id", "timestamp"], kind="stable")
    from_end = ordered.groupby("user_id").cumcount(ascending=False)
    size = ordered.groupby("user_id")["item_id"].transform("size")
    eligible = size >= min_events
    test = ordered[eligible & (from_end == 0)]
    val = ordered[eligible & (from_end == 1)]
    train = ordered[~(eligible & (from_end <= 1))]
    return train, val, test


def leave_one_out_check(
    ratings: pd.DataFrame,
    models: dict[str, ModelConfig],
    k: int,
    max_epochs: int = 200,
    checkpoint_dir: Path | None = None,
    monitor_users: int = 20_000,
) -> dict[str, Any]:
    """Fit each model on all but the last event per user and rank the last one.

    Models with early stopping (SASRec) choose their epoch count on the validation event
    first (up to ``max_epochs``; the count tuned for the temporal split does not carry
    over) and are then refit on train + validation, as in the temporal protocol.
    """
    train, val, test = leave_one_out_split(ratings)
    train_val = pd.concat([train, val])
    # Early stopping needs a stable signal, not every user: a fixed random sample keeps each
    # validation check cheap on large datasets (162k users on ml-25m). Final scores use all.
    users = val["user_id"].unique()
    if len(users) > monitor_users:
        keep = np.random.default_rng(0).choice(users, size=monitor_users, replace=False)
        val = val[val["user_id"].isin(keep)]
    out: dict[str, Any] = {}
    for name, mc in models.items():
        start = time.perf_counter()
        config = mc
        probe = build_model(mc.model_copy(update={"epochs": max_epochs}))
        if checkpoint_dir is not None and hasattr(probe, "set_checkpoint"):
            probe.set_checkpoint(checkpoint_dir / f"loo-probe-{name}.pt")
        if hasattr(probe, "set_monitor"):
            probe.set_monitor(lambda m: evaluate_ranking(m, val, k, ANY_INTERACTION)["all"]["ndcg"])
            probe.fit(train)
            config = mc.model_copy(update={"epochs": int(probe.metadata["best_epoch"])})
        del probe
        model = build_model(config)
        if checkpoint_dir is not None and hasattr(model, "set_checkpoint"):
            model.set_checkpoint(checkpoint_dir / f"loo-{name}.pt")
        model.fit(train_val)
        ranking = evaluate_ranking(model, test, k, ANY_INTERACTION)["all"]
        del model
        gc.collect()
        out[name] = {
            "ndcg": ranking["ndcg"],
            "hit_rate": ranking["hit_rate"],
            "n_users": ranking["n_users"],
            "fit_seconds": round(time.perf_counter() - start, 2),
        }
        logger.info(
            "leave-one-out %-12s ndcg@%d=%.4f hr@%d=%.4f",
            name,
            k,
            ranking["ndcg"],
            k,
            ranking["hit_rate"],
        )
    out["note"] = (
        "leave-one-out: each user's last event is the test item (any rating is relevant); "
        "training data includes events after other users' test events"
    )
    return out
