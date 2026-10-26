"""Feast feature definitions for the recommendation system.

* ``user_stats`` / ``item_stats``  - batch features computed from the training window by
  ``recsys features`` (parquet), materialized to the online store by ``recsys feast-materialize``.
* ``user_realtime``                - fresh features *pushed* by the Kafka stream processor
  (last-N interactions per user); read by the API on every request for real-time fold-in.

Paths are relative to this repo directory (Feast resolves them against ``repo_path``).
"""

from datetime import timedelta

from feast import (
    Entity,
    FeatureService,
    FeatureView,
    Field,
    FileSource,
    Project,
    PushSource,
    ValueType,
)
from feast.types import Array, Float32, Int64, String

project = Project(name="recsys", description="MovieLens recommendation features")

# A ttl of 0 means "never expires": MovieLens timestamps are historical (replayed), so
# wall-clock based expiry would hide every value. Production would use e.g. 30 days.
NO_EXPIRY = timedelta(0)

user = Entity(
    name="user",
    join_keys=["user_id"],
    value_type=ValueType.INT64,
    description="MovieLens userId",
)
movie = Entity(
    name="movie",
    join_keys=["item_id"],
    value_type=ValueType.INT64,
    description="MovieLens movieId",
)

# --------------------------------------------------------------------------- batch sources
user_stats_source = FileSource(
    name="user_stats_source",
    path="../data/features/user_features.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created",
)
item_stats_source = FileSource(
    name="item_stats_source",
    path="../data/features/item_features.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created",
)

user_stats = FeatureView(
    name="user_stats",
    entities=[user],
    ttl=NO_EXPIRY,
    schema=[
        Field(name="user_rating_count", dtype=Int64),
        Field(name="user_avg_rating", dtype=Float32),
        Field(name="user_rating_std", dtype=Float32),
        Field(name="user_days_active", dtype=Float32),
        Field(name="user_top_genre", dtype=String),
    ],
    source=user_stats_source,
    online=True,
    tags={"owner": "recsys", "freshness": "daily"},
)

item_stats = FeatureView(
    name="item_stats",
    entities=[movie],
    ttl=NO_EXPIRY,
    schema=[
        Field(name="item_rating_count", dtype=Int64),
        Field(name="item_avg_rating", dtype=Float32),
        Field(name="item_bayes_avg", dtype=Float32),
        Field(name="item_popularity_rank", dtype=Int64),
        Field(name="item_year", dtype=Int64),
        Field(name="item_genres", dtype=String),
    ],
    source=item_stats_source,
    online=True,
    tags={"owner": "recsys", "freshness": "daily"},
)

# --------------------------------------------------------------------------- streaming
user_realtime_push = PushSource(
    name="user_realtime_push",
    batch_source=FileSource(
        name="user_realtime_batch",
        path="../data/features/user_realtime.parquet",
        timestamp_field="event_timestamp",
    ),
)

user_realtime = FeatureView(
    name="user_realtime",
    entities=[user],
    ttl=NO_EXPIRY,
    schema=[
        Field(name="recent_items", dtype=Array(Int64)),
        Field(name="recent_ratings", dtype=Array(Float32)),
        Field(name="recent_ts", dtype=Array(Int64)),
        Field(name="rt_event_count", dtype=Int64),
        Field(name="last_event_ts", dtype=Int64),
    ],
    source=user_realtime_push,
    online=True,
    tags={"owner": "recsys", "freshness": "seconds"},
)

# --------------------------------------------------------------------------- services
realtime_service = FeatureService(name="recsys_realtime_v1", features=[user_realtime])
ranking_service = FeatureService(name="recsys_ranking_v1", features=[user_stats, item_stats])
