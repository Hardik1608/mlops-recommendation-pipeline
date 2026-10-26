"""Model package = everything the serving layer needs, in one directory.

::

    <package>/
      model/            Recommender.save() output (model.json + arrays.npz)
      catalog.parquet   item_id, title, genres  (only items the model can recommend)
      package.json      lineage: run id, dataset, split cutoffs, metrics, created_at

The same directory is logged to MLflow as the artifact of a pyfunc model, so a version in
the registry and a local export are byte-for-byte the same thing.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from recsys.models.base import Recommender
from recsys.models.io import load_model


@dataclass
class ModelPackage:
    model: Recommender
    catalog: dict[int, tuple[str, str]]
    info: dict[str, Any]

    @property
    def package_id(self) -> str:
        return str(self.info.get("package_id", "unknown"))

    @property
    def trained_through_ts(self) -> int:
        """Last event time included in training; newer events are real-time context."""
        return int(self.model.metadata.get("data_max_timestamp", 0))


def save_package(
    model: Recommender, movies: pd.DataFrame, out_dir: Path, info: dict[str, Any] | None = None
) -> Path:
    out_dir = Path(out_dir)
    model.save(out_dir / "model")
    catalog = movies.loc[
        movies["item_id"].isin(model.item_index.ids), ["item_id", "title", "genres"]
    ]
    catalog.to_parquet(out_dir / "catalog.parquet", index=False)
    digest = hashlib.sha256((out_dir / "model" / "arrays.npz").read_bytes()).hexdigest()
    payload = {
        "package_id": digest[:12],
        "model_type": model.model_type,
        "params": model.params,
        "created_at": datetime.now(tz=UTC).isoformat(),
        "model_metadata": model.metadata,
        **(info or {}),
    }
    (out_dir / "package.json").write_text(json.dumps(payload, indent=2, default=str), "utf-8")
    return out_dir


def load_package(path: str | Path) -> ModelPackage:
    path = Path(path)
    model = load_model(path / "model")
    catalog_df = pd.read_parquet(path / "catalog.parquet")
    catalog = {
        int(row.item_id): (str(row.title), str(row.genres))
        for row in catalog_df.itertuples(index=False)
    }
    info = json.loads((path / "package.json").read_text(encoding="utf-8"))
    return ModelPackage(model=model, catalog=catalog, info=info)
