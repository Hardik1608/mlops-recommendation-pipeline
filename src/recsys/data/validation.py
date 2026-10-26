"""Lightweight data-quality gate (the checks a Great Expectations suite would hold).

Hard failures raise ``DataValidationError`` so the Airflow task fails loudly before
anything is trained on bad data; soft findings are returned as warnings.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import pandas as pd

from recsys.data.loader import RATING_COLUMNS


class DataValidationError(ValueError):
    pass


@dataclass
class ValidationReport:
    n_ratings: int
    n_users: int
    n_items: int
    min_ts: int
    max_ts: int
    duplicate_user_item: int
    unknown_items: int
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def validate_ratings(
    ratings: pd.DataFrame,
    movies: pd.DataFrame | None = None,
    *,
    min_rating: float = 0.5,
    max_rating: float = 5.0,
    max_duplicate_fraction: float = 0.01,
) -> ValidationReport:
    missing = [c for c in RATING_COLUMNS if c not in ratings.columns]
    if missing:
        raise DataValidationError(f"missing columns: {missing}")
    if ratings.empty:
        raise DataValidationError("ratings table is empty")
    nulls = ratings[RATING_COLUMNS].isna().sum()
    if nulls.any():
        raise DataValidationError(f"null values found: {nulls[nulls > 0].to_dict()}")
    out_of_range = ~ratings["rating"].between(min_rating, max_rating)
    if out_of_range.any():
        raise DataValidationError(f"{int(out_of_range.sum())} ratings outside [{min_rating}, {max_rating}]")
    if (ratings["timestamp"] <= 0).any():
        raise DataValidationError("non-positive timestamps found")

    warnings: list[str] = []
    dupes = int(ratings.duplicated(["user_id", "item_id"]).sum())
    if dupes > max_duplicate_fraction * len(ratings):
        raise DataValidationError(f"{dupes} duplicate (user, item) pairs")
    if dupes:
        warnings.append(f"{dupes} duplicate (user, item) pairs (kept latest)")

    unknown = 0
    if movies is not None:
        unknown = int((~ratings["item_id"].isin(movies["item_id"])).sum())
        if unknown:
            warnings.append(f"{unknown} ratings reference items missing from the catalog")

    return ValidationReport(
        n_ratings=len(ratings),
        n_users=int(ratings["user_id"].nunique()),
        n_items=int(ratings["item_id"].nunique()),
        min_ts=int(ratings["timestamp"].min()),
        max_ts=int(ratings["timestamp"].max()),
        duplicate_user_item=dupes,
        unknown_items=unknown,
        warnings=warnings,
    )


def deduplicate_latest(ratings: pd.DataFrame) -> pd.DataFrame:
    """Keep the most recent rating per (user, item)."""
    return (
        ratings.sort_values("timestamp")
        .drop_duplicates(["user_id", "item_id"], keep="last")
        .reset_index(drop=True)
    )
