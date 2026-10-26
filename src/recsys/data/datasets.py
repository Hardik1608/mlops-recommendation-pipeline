"""Registry of supported MovieLens variants (plus a locally generated synthetic dataset)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

GROUPLENS = "https://files.grouplens.org/datasets/movielens"


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    url: str | None  # None => generated locally (synthetic)
    ratings_file: str
    movies_file: str
    fmt: Literal["csv", "dat"]
    sha256: str | None = None  # pinned for the variants verified in this repo


DATASETS: dict[str, DatasetSpec] = {
    "ml-latest-small": DatasetSpec(
        name="ml-latest-small",
        url=f"{GROUPLENS}/ml-latest-small.zip",
        ratings_file="ratings.csv",
        movies_file="movies.csv",
        fmt="csv",
        sha256="696d65a3dfceac7c45750ad32df2c259311949efec81f0f144fdfb91ebc9e436",
    ),
    "ml-1m": DatasetSpec(
        name="ml-1m",
        url=f"{GROUPLENS}/ml-1m.zip",
        ratings_file="ratings.dat",
        movies_file="movies.dat",
        fmt="dat",
        sha256="a6898adb50b9ca05aa231689da44c217cb524e7ebd39d264c56e2832f2c54e20",
    ),
    "ml-25m": DatasetSpec(
        name="ml-25m",
        url=f"{GROUPLENS}/ml-25m.zip",
        ratings_file="ratings.csv",
        movies_file="movies.csv",
        fmt="csv",
        sha256="8b21cfb7eb1706b4ec0aac894368d90acf26ebdfb6aced3ebd4ad5bd1eb9c6aa",
    ),
    # Shares the CSV layout; checksum not pinned (not downloaded in this repo).
    "ml-32m": DatasetSpec(
        name="ml-32m",
        url=f"{GROUPLENS}/ml-32m.zip",
        ratings_file="ratings.csv",
        movies_file="movies.csv",
        fmt="csv",
    ),
    "synthetic": DatasetSpec(
        name="synthetic",
        url=None,
        ratings_file="ratings.csv",
        movies_file="movies.csv",
        fmt="csv",
    ),
}


def get_spec(name: str) -> DatasetSpec:
    try:
        return DATASETS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown dataset {name!r}. Choose one of {sorted(DATASETS)}") from exc
