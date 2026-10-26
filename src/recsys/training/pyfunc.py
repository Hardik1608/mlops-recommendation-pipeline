"""MLflow pyfunc wrapper so registered versions are loadable by any MLflow tooling.

``mlflow.pyfunc.load_model("models:/recsys-ranker@champion").predict(df, params={"k": 5})``
returns a ranked list per user (batch scoring); the API unwraps the native package instead.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import mlflow.pyfunc
import pandas as pd

from recsys.models.base import Recommender
from recsys.models.package import ModelPackage, load_package


def recommend_frame(model: Recommender, user_ids: Iterable[int], k: int) -> pd.DataFrame:
    rows = []
    for user_id in user_ids:
        rec = model.recommend(int(user_id), k)
        rows.extend(
            {
                "user_id": int(user_id),
                "rank": rank,
                "item_id": int(item),
                "score": float(score),
                "strategy": rec.strategy,
            }
            for rank, (item, score) in enumerate(zip(rec.items, rec.scores, strict=True), 1)
        )
    return pd.DataFrame(rows, columns=["user_id", "rank", "item_id", "score", "strategy"])


class RecsysPyFunc(mlflow.pyfunc.PythonModel):
    package: ModelPackage

    def load_context(self, context: Any) -> None:
        self.package = load_package(context.artifacts["package"])

    def predict(self, context: Any, model_input: pd.DataFrame, params: dict | None = None):
        k = int((params or {}).get("k", 10))
        return recommend_frame(self.package.model, model_input["user_id"].astype("int64"), k)
