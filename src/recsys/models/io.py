"""Model factory + loading of saved model directories."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from recsys.config import ModelConfig
from recsys.models.als import ExplicitALS, ImplicitALS
from recsys.models.base import FORMAT_VERSION, Recommender
from recsys.models.popularity import PopularityRecommender
from recsys.models.sasrec import SASRec

MODEL_CLASSES: dict[str, type[Recommender]] = {
    cls.model_type: cls for cls in (PopularityRecommender, ExplicitALS, ImplicitALS, SASRec)
}

SASREC_FIELDS = (
    "max_len",
    "num_blocks",
    "num_heads",
    "dropout",
    "learning_rate",
    "batch_size",
    "epochs",
    "loss",
    "num_negatives",
    "eval_every",
    "patience",
    "device",
    "sequence_order",
)


def model_params(config: ModelConfig) -> dict[str, Any]:
    """Only the hyper-parameters relevant to the chosen model type."""
    if config.type == "popularity":
        return {"popularity_half_life_days": config.popularity_half_life_days}
    if config.type == "sasrec":
        return {
            "factors": config.factors,
            "seed": config.seed,
            "popularity_half_life_days": config.popularity_half_life_days,
            **{name: getattr(config, name) for name in SASREC_FIELDS},
        }
    params: dict[str, Any] = {
        "factors": config.factors,
        "regularization": config.regularization,
        "iterations": config.iterations,
        "seed": config.seed,
        # popularity also drives the cold-start fallback of every model
        "popularity_half_life_days": config.popularity_half_life_days,
    }
    if config.backend != "numpy":  # recorded only when used, so existing runs keep their params
        params.update(backend=config.backend, device=config.device)
    if config.type == "implicit_als":
        params.update(
            alpha=config.alpha,
            weighting=config.weighting,
            recency_half_life_days=config.recency_half_life_days,
            solver=config.solver,
            cg_steps=config.cg_steps,
            blend_weight=config.blend_weight,
            blend_half_life_days=config.blend_half_life_days,
        )
    return params


def build_model(config: ModelConfig) -> Recommender:
    return MODEL_CLASSES[config.type](**model_params(config))


def load_model(path: str | Path) -> Recommender:
    path = Path(path)
    meta = json.loads((path / "model.json").read_text(encoding="utf-8"))
    if meta.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"unsupported model format {meta.get('format_version')!r}")
    with np.load(path / "arrays.npz") as npz:
        arrays = {name: npz[name] for name in npz.files}
    return MODEL_CLASSES[meta["model_type"]].from_saved(meta, arrays)
