"""Batch features computed from the training window (registered in Feast, see feature_repo/).

All rows carry ``event_timestamp`` = the snapshot time (training cutoff), which is what makes
Feast's point-in-time joins correct: a training example at time t only sees feature values
computed from data before t.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from recsys.data.loader import extract_year

logger = logging.getLogger(__name__)

REALTIME_SCHEMA = pa.schema(
    [
        ("user_id", pa.int64()),
        ("event_timestamp", pa.timestamp("us", tz="UTC")),
        ("recent_items", pa.list_(pa.int64())),
        ("recent_ratings", pa.list_(pa.float32())),
        ("recent_ts", pa.list_(pa.int64())),
        ("rt_event_count", pa.int64()),
        ("last_event_ts", pa.int64()),
    ]
)


def _snapshot(as_of: int) -> pd.Timestamp:
    return pd.Timestamp(as_of, unit="s", tz="UTC")


def build_user_features(ratings: pd.DataFrame, movies: pd.DataFrame, as_of: int) -> pd.DataFrame:
    grouped = ratings.groupby("user_id")
    users = pd.DataFrame(
        {
            "user_rating_count": grouped.size().astype("int64"),
            "user_avg_rating": grouped["rating"].mean().astype("float32"),
            "user_rating_std": grouped["rating"].std().fillna(0.0).astype("float32"),
            "user_days_active": (
                (grouped["timestamp"].max() - grouped["timestamp"].min()) / 86_400
            ).astype("float32"),
        }
    )
    liked = ratings.loc[ratings["rating"] >= 4.0, ["user_id", "item_id"]].merge(
        movies[["item_id", "genres"]], on="item_id", how="left"
    )
    liked = liked.assign(genre=liked["genres"].fillna("").str.split("|")).explode("genre")
    liked = liked[liked["genre"].ne("") & liked["genre"].notna()]
    top = (
        liked.groupby(["user_id", "genre"])
        .size()
        .rename("n")
        .reset_index()
        .sort_values(["user_id", "n", "genre"], ascending=[True, False, True])
        .drop_duplicates("user_id")
        .set_index("user_id")["genre"]
    )
    users["user_top_genre"] = top.reindex(users.index).fillna("unknown").astype(str)
    users = users.reset_index()
    users["event_timestamp"] = _snapshot(as_of)
    return users


def build_item_features(
    ratings: pd.DataFrame, movies: pd.DataFrame, as_of: int, prior_weight: float = 25.0
) -> pd.DataFrame:
    grouped = ratings.groupby("item_id")["rating"]
    stats = pd.DataFrame({"count": grouped.size(), "mean": grouped.mean()})
    items = movies[["item_id", "title", "genres"]].merge(
        stats, left_on="item_id", right_index=True, how="left"
    )
    global_mean = float(ratings["rating"].mean())
    count = items["count"].fillna(0)
    mean = items["mean"].fillna(global_mean)
    items["item_rating_count"] = count.astype("int64")
    items["item_avg_rating"] = mean.astype("float32")
    items["item_bayes_avg"] = (
        (count * mean + prior_weight * global_mean) / (count + prior_weight)
    ).astype("float32")
    items["item_popularity_rank"] = count.rank(ascending=False, method="first").astype("int64")
    items["item_year"] = items["title"].map(extract_year).fillna(0).astype("int64")
    items = items.rename(columns={"genres": "item_genres"})
    items["event_timestamp"] = _snapshot(as_of)
    return items[
        [
            "item_id",
            "event_timestamp",
            "item_rating_count",
            "item_avg_rating",
            "item_bayes_avg",
            "item_popularity_rank",
            "item_year",
            "item_genres",
        ]
    ]


def write_feature_tables(
    ratings: pd.DataFrame, movies: pd.DataFrame, out_dir: Path, as_of: int | None = None
) -> dict[str, Path]:
    """Write the Feast batch sources and an empty-schema real-time source."""
    out_dir.mkdir(parents=True, exist_ok=True)
    as_of = as_of if as_of is not None else int(ratings["timestamp"].max())
    created = pd.Timestamp(datetime.now(tz=UTC))

    users = build_user_features(ratings, movies, as_of).assign(created=created)
    items = build_item_features(ratings, movies, as_of).assign(created=created)
    paths = {
        "user_features": out_dir / "user_features.parquet",
        "item_features": out_dir / "item_features.parquet",
        "user_realtime": out_dir / "user_realtime.parquet",
    }
    users.to_parquet(paths["user_features"], index=False)
    items.to_parquet(paths["item_features"], index=False)
    if not paths["user_realtime"].exists():
        pq.write_table(REALTIME_SCHEMA.empty_table(), paths["user_realtime"])
    logger.info(
        "Wrote %d user and %d item feature rows (snapshot ts=%d) to %s",
        len(users),
        len(items),
        as_of,
        out_dir,
    )
    return paths
