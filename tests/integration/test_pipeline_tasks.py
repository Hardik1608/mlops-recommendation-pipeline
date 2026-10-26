"""The functions Airflow calls, exercised without Airflow."""

from __future__ import annotations

import json

import pytest
from tests.conftest import make_settings

from recsys.cli import main as cli_main
from recsys.pipelines import tasks

pytestmark = pytest.mark.integration


def test_ingest_split_features_chain(tmp_path) -> None:
    settings = make_settings(tmp_path)
    raw = tasks.ingest("synthetic", settings)  # generated locally, never downloaded
    assert (tmp_path / "data" / "raw" / "synthetic" / "ratings.csv").exists() and raw
    info = tasks.validate_and_split("synthetic", settings)
    assert info["n_train"] > info["n_test"] > 0
    paths = tasks.build_features("synthetic", settings)
    assert set(paths) == {"user_features", "item_features", "user_realtime"}


def test_retrain_trigger() -> None:
    assert tasks.should_retrain({"drift_detected": True})
    assert not tasks.should_retrain({"drift_detected": False})
    assert not tasks.should_retrain({})


def test_cli_prepare_and_stream_without_broker(tmp_path, monkeypatch, capsys) -> None:
    settings = make_settings(tmp_path)
    for key, value in {
        "RECSYS_DATA_DIR": settings.data_dir,
        "RECSYS_CONFIGS_DIR": settings.configs_dir,
        "RECSYS_REPORTS_DIR": settings.reports_dir,
        "RECSYS_DATASET": "synthetic",
        "RECSYS_CACHE_BACKEND": "none",
    }.items():
        monkeypatch.setenv(key, str(value))
    assert cli_main(["prepare"]) == 0
    assert cli_main(["stream", "--limit", "120"]) == 0
    out = capsys.readouterr().out
    summary = json.loads(out[out.rindex('{\n  "published"') :])
    assert summary["published"] == summary["processed"] == 120
    assert list((settings.data_dir / "events").glob("date=*/*.parquet"))
