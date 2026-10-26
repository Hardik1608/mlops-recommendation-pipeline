"""Deterministic, MovieLens-shaped synthetic data for tests and CI (no downloads).

The generator plants real structure so that models can be meaningfully compared:
genre-clustered latent item vectors, user taste vectors, a Zipf popularity prior,
users that arrive over time (=> cold-start users in the test window) and ratings
that depend on user/item affinity.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

GENRES = [
    "Action",
    "Adventure",
    "Animation",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Horror",
    "Romance",
    "Sci-Fi",
    "Thriller",
]
START_TS = 1_420_070_400  # 2015-01-01
DAY = 86_400


def generate_synthetic(
    n_users: int = 300,
    n_items: int = 500,
    n_factors: int = 8,
    mean_ratings_per_user: float = 30.0,
    period_days: int = 720,
    seed: int = 7,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(ratings, movies)`` in the raw MovieLens CSV column layout."""
    rng = np.random.default_rng(seed)
    n_genres = len(GENRES)

    centers = rng.normal(0.0, 1.0, size=(n_genres, n_factors))
    item_genre = rng.integers(0, n_genres, size=n_items)
    item_vecs = centers[item_genre] + 0.6 * rng.normal(size=(n_items, n_factors))
    item_bias = rng.normal(0.0, 0.3, size=n_items)
    popularity = 1.0 / np.power(rng.permutation(n_items) + 1.0, 0.8)

    fav_genre = rng.integers(0, n_genres, size=n_users)
    user_vecs = centers[fav_genre] + 0.8 * rng.normal(size=(n_users, n_factors))
    arrival = np.sort(rng.uniform(0, period_days * 0.95, size=n_users))
    activity_span = rng.uniform(5, period_days * 0.4, size=n_users)

    # Non-contiguous raw ids exercise the id <-> index mapping in the models.
    item_ids = 10 * np.arange(1, n_items + 1) + 1
    user_ids = np.arange(1, n_users + 1)

    rows: list[tuple[int, int, float, int]] = []
    for u in range(n_users):
        n_u = int(np.clip(rng.lognormal(np.log(mean_ratings_per_user), 0.6), 5, n_items // 2))
        affinity = item_vecs @ user_vecs[u] / np.sqrt(n_factors)
        logits = np.log(popularity) + 1.5 * affinity
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        items = rng.choice(n_items, size=n_u, replace=False, p=probs)
        raw = 3.4 + 0.9 * affinity[items] + item_bias[items] + rng.normal(0, 0.5, size=n_u)
        ratings = np.clip(np.round(raw * 2) / 2, 0.5, 5.0)
        offsets = np.sort(rng.uniform(0, activity_span[u], size=n_u))
        ts = START_TS + ((arrival[u] + offsets) * DAY).astype(np.int64)
        rows.extend(
            (int(user_ids[u]), int(item_ids[i]), float(r), int(t))
            for i, r, t in zip(items, ratings, ts, strict=True)
        )

    ratings_df = pd.DataFrame(rows, columns=["userId", "movieId", "rating", "timestamp"])
    ratings_df = ratings_df.sort_values(["timestamp", "userId"]).reset_index(drop=True)

    second_genre = rng.integers(0, n_genres, size=n_items)
    years = rng.integers(1970, 2015, size=n_items)
    movies_df = pd.DataFrame(
        {
            "movieId": item_ids,
            "title": [f"Synthetic Movie {i} ({years[i]})" for i in range(n_items)],
            "genres": [
                GENRES[g] if g == s else f"{GENRES[g]}|{GENRES[s]}"
                for g, s in zip(item_genre, second_genre, strict=True)
            ],
        }
    )
    return ratings_df, movies_df


def write_synthetic(out_dir: Path, **kwargs: int | float) -> Path:
    """Write ``ratings.csv`` + ``movies.csv`` to ``out_dir`` and return it."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ratings, movies = generate_synthetic(**kwargs)  # type: ignore[arg-type]
    ratings.to_csv(out_dir / "ratings.csv", index=False)
    movies.to_csv(out_dir / "movies.csv", index=False)
    return out_dir
