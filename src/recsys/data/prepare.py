"""Raw files -> validated, de-duplicated, temporally split parquet tables.

Output (``data/processed/<dataset>/``): ``train|val|test.parquet``, ``movies.parquet`` and
``split.json`` (cutoffs, counts, raw-file checksum for lineage).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from recsys.config import Settings, SplitConfig
from recsys.data.datasets import get_spec
from recsys.data.download import sha256_file
from recsys.data.loader import load_dataset
from recsys.data.split import TemporalSplit, temporal_split
from recsys.data.validation import deduplicate_latest, validate_ratings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreparedData:
    split: TemporalSplit
    movies: pd.DataFrame
    info: dict


def prepare_dataset(settings: Settings, dataset: str, split_cfg: SplitConfig) -> PreparedData:
    raw_dir = settings.raw_dir / dataset
    ratings, movies = load_dataset(settings.raw_dir, dataset)
    report = validate_ratings(ratings, movies)
    for warning in report.warnings:
        logger.warning("data quality: %s", warning)
    ratings = deduplicate_latest(ratings)
    split = temporal_split(ratings, split_cfg.val_fraction, split_cfg.test_fraction)

    out = settings.processed_dir(dataset)
    out.mkdir(parents=True, exist_ok=True)
    split.train.to_parquet(out / "train.parquet", index=False)
    split.val.to_parquet(out / "val.parquet", index=False)
    split.test.to_parquet(out / "test.parquet", index=False)
    movies.to_parquet(out / "movies.parquet", index=False)

    info = {
        "dataset": dataset,
        "ratings_sha256": sha256_file(raw_dir / get_spec(dataset).ratings_file),
        "validation": report.to_dict(),
        **split.summary(),
        "val_fraction": split_cfg.val_fraction,
        "test_fraction": split_cfg.test_fraction,
    }
    (out / "split.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    logger.info(
        "Prepared %s: train=%d val=%d test=%d -> %s",
        dataset,
        len(split.train),
        len(split.val),
        len(split.test),
        out,
    )
    return PreparedData(split=split, movies=movies, info=info)


def load_prepared(settings: Settings, dataset: str) -> PreparedData:
    out = settings.processed_dir(dataset)
    info_path = out / "split.json"
    if not info_path.exists():
        raise FileNotFoundError(
            f"{info_path} not found - run `recsys prepare --dataset {dataset}` first"
        )
    info = json.loads(info_path.read_text(encoding="utf-8"))
    split = TemporalSplit(
        train=pd.read_parquet(out / "train.parquet"),
        val=pd.read_parquet(out / "val.parquet"),
        test=pd.read_parquet(out / "test.parquet"),
        val_start=int(info["val_start"]),
        test_start=int(info["test_start"]),
    )
    return PreparedData(split=split, movies=pd.read_parquet(out / "movies.parquet"), info=info)


def ensure_raw(settings: Settings, dataset: str) -> Path:
    """Make sure the raw files exist (download, or generate for ``synthetic``)."""
    spec = get_spec(dataset)
    target = settings.raw_dir / dataset
    if (target / spec.ratings_file).exists():
        return target
    if spec.url is None:
        from recsys.data.synthetic import write_synthetic

        return write_synthetic(target)
    from recsys.data.download import download_dataset

    return download_dataset(dataset, settings.raw_dir)
