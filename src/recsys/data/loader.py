"""Load raw MovieLens files into a normalised schema.

ratings: ``user_id:int64, item_id:int64, rating:float32, timestamp:int64`` (unix seconds)
movies:  ``item_id:int64, title:str, genres:str`` (pipe-separated, e.g. ``"Action|Comedy"``)
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from recsys.data.datasets import get_spec

RATING_COLUMNS = ["user_id", "item_id", "rating", "timestamp"]
MOVIE_COLUMNS = ["item_id", "title", "genres"]
_YEAR_RE = re.compile(r"\((\d{4})\)\s*$")


def _normalise_ratings(df: pd.DataFrame) -> pd.DataFrame:
    return df.astype(
        {"user_id": "int64", "item_id": "int64", "rating": "float32", "timestamp": "int64"}
    )[RATING_COLUMNS]


def load_ratings(dataset_dir: Path, dataset: str) -> pd.DataFrame:
    spec = get_spec(dataset)
    path = Path(dataset_dir) / spec.ratings_file
    if spec.fmt == "csv":
        df = pd.read_csv(path)
        df = df.rename(columns={"userId": "user_id", "movieId": "item_id"})
    else:
        # ``::``-separated .dat files: splitting on ':' yields empty columns in between,
        # which keeps pandas on the fast C parser.
        df = pd.read_csv(path, sep=":", header=None, usecols=[0, 2, 4, 6], engine="c")
        df.columns = RATING_COLUMNS
    return _normalise_ratings(df)


def load_movies(dataset_dir: Path, dataset: str) -> pd.DataFrame:
    spec = get_spec(dataset)
    path = Path(dataset_dir) / spec.movies_file
    if spec.fmt == "csv":
        df = pd.read_csv(path).rename(columns={"movieId": "item_id"})
    else:  # titles can contain ':' so the python engine is needed for movies.dat
        df = pd.read_csv(
            path, sep="::", header=None, names=MOVIE_COLUMNS, engine="python", encoding="latin-1"
        )
    df = df[MOVIE_COLUMNS].copy()
    df["item_id"] = df["item_id"].astype("int64")
    df["title"] = df["title"].astype(str)
    df["genres"] = df["genres"].fillna("(no genres listed)").astype(str)
    return df


def extract_year(title: str) -> int | None:
    match = _YEAR_RE.search(title)
    return int(match.group(1)) if match else None


def load_dataset(raw_dir: Path, dataset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset_dir = Path(raw_dir) / dataset
    return load_ratings(dataset_dir, dataset), load_movies(dataset_dir, dataset)
