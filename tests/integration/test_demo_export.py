"""Static-site demo export: valid JSON, consistent with the models it describes."""

from __future__ import annotations

import json

import pytest

from recsys.evaluation.demo_export import export_demo

pytestmark = pytest.mark.integration


def test_export_demo_writes_self_contained_json(settings, prepared, tmp_path) -> None:
    out = tmp_path / "demo.json"
    payload = export_demo(settings, "synthetic", out, n_users=4, k=5)
    assert json.loads(out.read_text(encoding="utf-8")) == payload
    assert payload["models"] == ["popularity", "hybrid_als"]
    assert 0 < len(payload["users"]) <= 4
    for user in payload["users"]:
        assert len(user["streamed"]) == payload["context_events"]
        for name in payload["models"]:
            recs = user["recs"][name]
            assert len(recs) == 5 and not set(recs) & set(user["history"])
            assert not set(user["recs_streaming"][name]) & set(user["streamed"])
        mentioned = user["history"] + user["liked_later"] + user["recs"]["hybrid_als"]
        assert all(str(item) in payload["movies"] for item in mentioned)
