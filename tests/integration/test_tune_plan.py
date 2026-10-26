"""Staged tuning plans: later stages keep earlier winners fixed; plans can be resumed."""

from __future__ import annotations

import json

import pytest
import yaml

from recsys.training.tune import run_plan

pytestmark = pytest.mark.integration


@pytest.fixture()
def plan(tmp_path, prepared):
    stages = [
        {
            "name": "p1_popularity",
            "model": "popularity",
            "metric": "ndcg_at_10",
            "grid": {"popularity_half_life_days": [None, 7]},
        },
        {
            "name": "p2_als_alpha",
            "model": "implicit_als",
            "metric": "warm_ndcg_at_10",
            "grid": {"alpha": [1, 5], "iterations": [3]},
        },
        {
            "name": "p3_als_factors",
            "model": "implicit_als",
            "metric": "warm_ndcg_at_10",
            "grid": {"factors": [8, 16]},
        },
    ]
    path = tmp_path / "plan.yaml"
    path.write_text(yaml.safe_dump({"dataset": "synthetic", "stages": stages}), "utf-8")
    return path


def test_later_stages_keep_earlier_winners(settings, plan) -> None:
    selected = run_plan(settings, plan)
    stage2 = json.loads((settings.reports_dir / "tuning/synthetic/p2_als_alpha.json").read_text())
    stage3 = json.loads((settings.reports_dir / "tuning/synthetic/p3_als_factors.json").read_text())
    best_alpha = stage2["best"]["params"]["alpha"]
    assert all(r["params"]["alpha"] == best_alpha for r in stage3["results"])
    assert all(r["params"]["iterations"] == 3 for r in stage3["results"])
    assert selected["implicit_als"]["alpha"] == best_alpha
    assert selected["implicit_als"]["factors"] == stage3["best"]["params"]["factors"]
    assert "popularity_half_life_days" in selected["popularity"]
    assert selected["implicit_als"]["selected_by"].startswith("p3_als_factors")
    saved = settings.reports_dir / "tuning/synthetic/selected_plan.json"
    assert json.loads(saved.read_text()) == selected


def test_resume_reads_finished_stages_back(settings, plan) -> None:
    first = run_plan(settings, plan)
    stage2_file = settings.reports_dir / "tuning/synthetic/p2_als_alpha.json"
    before = stage2_file.stat().st_mtime_ns
    resumed = run_plan(settings, plan, start=3)
    assert stage2_file.stat().st_mtime_ns == before  # stage 2 was not re-run
    assert resumed["implicit_als"]["alpha"] == first["implicit_als"]["alpha"]


def test_an_interrupted_stage_reuses_its_finished_trials(settings, monkeypatch) -> None:
    from recsys.models import ImplicitALS
    from recsys.training.tune import run_grid

    grid = {"alpha": [1, 5, 20], "iterations": [2]}
    fits = {"n": 0}
    real_fit = ImplicitALS.fit

    def fit_then_crash(self, ratings):
        fits["n"] += 1
        if fits["n"] == 3:
            raise RuntimeError("power cut")
        return real_fit(self, ratings)

    monkeypatch.setattr(ImplicitALS, "fit", fit_then_crash)
    with pytest.raises(RuntimeError, match="power cut"):
        run_grid(settings, "synthetic", "implicit_als", grid, "warm_ndcg_at_10", name="resume")
    partial = settings.reports_dir / "tuning/synthetic/resume.partial.json"
    assert len(json.loads(partial.read_text())["results"]) == 2

    monkeypatch.setattr(ImplicitALS, "fit", real_fit)
    fits["n"] = 0
    calls = {"n": 0}

    def counting_fit(self, ratings):
        calls["n"] += 1
        return real_fit(self, ratings)

    monkeypatch.setattr(ImplicitALS, "fit", counting_fit)
    summary = run_grid(
        settings, "synthetic", "implicit_als", grid, "warm_ndcg_at_10", name="resume"
    )
    assert calls["n"] == 1  # only the unfinished trial ran
    assert len(summary["results"]) == 3
    assert not partial.exists()


def test_a_configuration_from_an_earlier_stage_is_not_trained_again(
    settings, tmp_path, monkeypatch
) -> None:
    from recsys.models import ImplicitALS

    # Stage 2 re-enters stage 1's grid point factors=16 (the synthetic base config).
    stages = [
        {
            "name": "r1",
            "model": "implicit_als",
            "metric": "warm_ndcg_at_10",
            "grid": {"alpha": [5.0], "iterations": [2]},
        },
        {
            "name": "r2",
            "model": "implicit_als",
            "metric": "warm_ndcg_at_10",
            "grid": {"factors": [16, 8]},
        },
    ]
    plan = tmp_path / "reuse.yaml"
    plan.write_text(yaml.safe_dump({"dataset": "synthetic", "stages": stages}), "utf-8")
    calls = {"n": 0}
    real_fit = ImplicitALS.fit

    def counting_fit(self, ratings):
        calls["n"] += 1
        return real_fit(self, ratings)

    monkeypatch.setattr(ImplicitALS, "fit", counting_fit)
    run_plan(settings, plan)
    assert calls["n"] == 2  # stage 2 trained only factors=8
    calls["n"] = 0
    run_plan(settings, plan, start=2)  # stage 1 read back still feeds the reuse table
    assert calls["n"] == 1
