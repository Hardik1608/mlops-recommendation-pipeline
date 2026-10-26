"""Export precomputed recommendations for the static portfolio demo (no server needed).

For a seeded sample of test users: the most recent training-window history, the movies the
user went on to rate >= threshold in the test window ("what they watched next"), and every
model's top-K before and after the user's first ``context_events`` test events arrive as
real-time context. Models are fit on train + validation with the tuned configs, exactly as
in the benchmark, so the demo shows the same models the README reports.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from recsys.config import Settings, load_train_config
from recsys.data.prepare import load_prepared
from recsys.models.io import build_model

logger = logging.getLogger(__name__)


def _title_map(movies: pd.DataFrame) -> dict[int, list[str]]:
    return {
        int(row.item_id): [str(row.title), str(row.genres)]
        for row in movies.itertuples(index=False)
    }


def export_demo(
    settings: Settings,
    dataset: str,
    out: Path,
    n_users: int = 60,
    k: int = 10,
    history_shown: int = 8,
    seed: int = 7,
    include_sequential: bool = True,
) -> dict[str, Any]:
    cfg = load_train_config(settings, dataset)
    prepared = load_prepared(settings, dataset)
    train, test = prepared.split.train_val, prepared.split.test
    threshold, n_context = cfg.evaluation.relevance_threshold, cfg.evaluation.context_events

    configs = {"popularity": cfg.baseline, "hybrid_als": cfg.model}
    if include_sequential and cfg.sequential is not None:  # SASRec needs torch (GPU)
        configs["sasrec"] = cfg.sequential.model_copy(
            update={"popularity_half_life_days": cfg.baseline.popularity_half_life_days}
        )
    models = {}
    for name, mc in configs.items():
        logger.info("fitting %s for the demo", name)
        models[name] = build_model(mc).fit(train)

    # Users the demo can tell a story about: known to the models, with a real history and
    # enough events after the cutoff to both feed the stream and leave hits to find.
    seen_counts = train.groupby("user_id").size()
    ordered_test = test.sort_values(["user_id", "timestamp"], kind="stable")
    candidates = []
    for user, events in ordered_test.groupby("user_id", sort=False):
        rest = events.iloc[n_context:]
        if (
            seen_counts.get(user, 0) >= history_shown
            and len(rest) >= k
            and (rest["rating"] >= threshold).sum() >= 3
        ):
            candidates.append(int(user))
    rng = np.random.default_rng(seed)
    picked = rng.choice(candidates, size=min(n_users, len(candidates)), replace=False)
    chosen = sorted(int(user) for user in picked)

    titles = _title_map(prepared.movies)
    ordered_train = train.sort_values(["user_id", "timestamp"], kind="stable")
    history_by_user = {
        int(u): g["item_id"].tolist()[-history_shown:]
        for u, g in ordered_train[ordered_train["user_id"].isin(chosen)].groupby("user_id")
    }
    users = []
    used_items: set[int] = set()
    for user in chosen:
        events = ordered_test[ordered_test["user_id"] == user]
        context, rest = events.iloc[:n_context], events.iloc[n_context:]
        liked_later = rest.loc[rest["rating"] >= threshold, "item_id"].astype(int).tolist()
        entry: dict[str, Any] = {
            "user_id": user,
            "history": history_by_user[user],
            "streamed": context["item_id"].astype(int).tolist(),
            "liked_later": liked_later,
            "recs": {},
            "recs_streaming": {},
        }
        for name, model in models.items():
            entry["recs"][name] = model.recommend(user, k).items
            entry["recs_streaming"][name] = model.recommend(
                user,
                k,
                context_items=entry["streamed"],
                context_ratings=context["rating"].tolist(),
            ).items
        users.append(entry)
        used_items.update(entry["history"], entry["streamed"], liked_later)
        for recs in (*entry["recs"].values(), *entry["recs_streaming"].values()):
            used_items.update(recs)

    payload = {
        "dataset": dataset,
        "k": k,
        "relevance_threshold": threshold,
        "context_events": n_context,
        "test_start": int(prepared.split.test_start),
        "models": list(models),
        "users": users,
        "movies": {str(i): titles.get(i, [f"Movie {i}", ""]) for i in sorted(used_items)},
    }
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    logger.info("demo export: %d users, %d movies -> %s", len(users), len(used_items), out)
    return payload
