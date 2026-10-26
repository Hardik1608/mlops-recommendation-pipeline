import pandas as pd
import pytest

from recsys.data.datasets import get_spec
from recsys.data.loader import extract_year, load_dataset
from recsys.data.split import temporal_split
from recsys.data.synthetic import generate_synthetic
from recsys.data.validation import DataValidationError, deduplicate_latest, validate_ratings


def test_synthetic_is_deterministic() -> None:
    a, _ = generate_synthetic(n_users=50, n_items=80, seed=3)
    b, _ = generate_synthetic(n_users=50, n_items=80, seed=3)
    pd.testing.assert_frame_equal(a, b)


def test_synthetic_shape_and_ranges(settings) -> None:
    ratings, movies = load_dataset(settings.raw_dir, "synthetic")
    assert list(ratings.columns) == ["user_id", "item_id", "rating", "timestamp"]
    assert ratings["rating"].between(0.5, 5.0).all()
    assert set(ratings["item_id"]).issubset(set(movies["item_id"]))
    assert ratings.duplicated(["user_id", "item_id"]).sum() == 0


def test_temporal_split_has_no_leakage(prepared) -> None:
    split = prepared.split
    assert split.train["timestamp"].max() < split.val_start <= split.val["timestamp"].min()
    assert split.val["timestamp"].max() < split.test_start <= split.test["timestamp"].min()
    total = len(split.train) + len(split.val) + len(split.test)
    assert total == prepared.info["validation"]["n_ratings"]


def test_temporal_split_creates_cold_start_users(prepared) -> None:
    split = prepared.split
    known = set(split.train_val["user_id"])
    test_users = set(split.test["user_id"])
    assert test_users - known, "synthetic users arrive over time -> some test users are new"
    assert test_users & known, "and some are returning users"


def test_split_rejects_bad_fractions(ratings) -> None:
    with pytest.raises(ValueError):
        temporal_split(ratings, val_fraction=0.6, test_fraction=0.5)


def test_validation_catches_bad_data(ratings) -> None:
    bad = ratings.copy()
    bad.loc[0, "rating"] = 9.0
    with pytest.raises(DataValidationError):
        validate_ratings(bad)
    with pytest.raises(DataValidationError):
        validate_ratings(ratings.drop(columns="timestamp"))
    report = validate_ratings(ratings)
    assert report.n_ratings == len(ratings) and not report.warnings


def test_deduplicate_keeps_latest() -> None:
    frame = pd.DataFrame(
        {"user_id": [1, 1], "item_id": [5, 5], "rating": [2.0, 4.5], "timestamp": [10, 20]}
    )
    out = deduplicate_latest(frame)
    assert len(out) == 1 and out.iloc[0]["rating"] == 4.5


def test_dataset_registry_and_year_parsing() -> None:
    assert get_spec("ml-latest-small").sha256
    assert get_spec("ml-1m").fmt == "dat"
    with pytest.raises(ValueError):
        get_spec("ml-nope")
    assert extract_year("Toy Story (1995)") == 1995
    assert extract_year("No year") is None
