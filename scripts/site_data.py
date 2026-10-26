"""Build the portfolio site's results JSON from the committed benchmark reports.

    python scripts/site_data.py ../06_portfolio-site/data/recommender-results.json

Reads ``reports/benchmark_<dataset>.json`` (and, when a model is missing there, the
partial reports ``benchmark_<dataset>_<models>.json``) so every number on the site comes
from a report in this repository. Datasets without a report are skipped.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPORTS = Path(__file__).resolve().parents[1] / "reports"
DATASETS = {"ml-1m": "MovieLens 1M", "ml-25m": "MovieLens 25M"}
MODELS = ("popularity", "implicit_als", "hybrid_als", "sasrec", "sasrec_shuffled")


def _merged(dataset: str) -> dict | None:
    full = REPORTS / f"benchmark_{dataset}.json"
    if not full.exists():
        return None
    report = json.loads(full.read_text(encoding="utf-8"))
    for extra in sorted(REPORTS.glob(f"benchmark_{dataset}_*.json")):
        part = json.loads(extra.read_text(encoding="utf-8"))
        for name, result in part["models"].items():
            report["models"].setdefault(name, result)
        for name, result in (part.get("streaming_value") or {}).items():
            report["streaming_value"].setdefault(name, result)
        loo, part_loo = report.get("leave_one_out_check"), part.get("leave_one_out_check")
        if part_loo:
            report["leave_one_out_check"] = {**part_loo, **(loo or {})}
        for name, result in (part.get("significance") or {}).items():
            report.setdefault("significance", {}).setdefault(name, result)
    return report


def summarise(dataset: str, report: dict) -> dict:
    models = report["models"]
    loo = report.get("leave_one_out_check") or {}
    streaming = report.get("streaming_value") or {}
    protocol = report["protocol"]
    any_ranking = next(iter(models.values()))["ranking"]
    out = {
        "label": DATASETS[dataset],
        "test_users": any_ranking["all"]["n_users"],
        "returning_users": any_ranking["warm"]["n_users"],
        "new_users": any_ranking["cold"]["n_users"],
        "n_train_val": protocol["n_train_val"],
        "n_test": protocol["n_test"],
        "temporal": {},
        "loo": {},
        "streaming_warm": {},
        "significance": report.get("significance") or {},
        "training": {},
    }
    for name in MODELS:
        if name in models:
            ranking = models[name]["ranking"]
            # Models are compared on returning users: new users get the same popularity
            # fallback from every model, so including them only dilutes the differences.
            out["temporal"][name] = {
                "ndcg": ranking["warm"]["ndcg"],
                "ndcg_all": ranking["all"]["ndcg"],
                "recall_warm": ranking["warm"]["recall"],
                "coverage": ranking["all"]["coverage"],
                "fit_seconds": models[name]["fit_seconds"],
            }
            if models[name].get("training"):
                out["training"][name] = models[name]["training"]
        if name in loo:
            out["loo"][name] = {"ndcg": loo[name]["ndcg"], "hit_rate": loo[name]["hit_rate"]}
        if name in streaming:
            warm = streaming[name]["warm"]
            out["streaming_warm"][name] = {
                "batch": warm["batch_only"]["ndcg"],
                "streaming": warm["streaming"]["ndcg"],
                "users": warm["batch_only"]["n_users"],
            }
    return out


def main(target: str) -> None:
    site = {"source": "reports/benchmark_<dataset>*.json", "datasets": {}}
    for dataset in DATASETS:
        report = _merged(dataset)
        if report is not None:
            site["datasets"][dataset] = summarise(dataset, report)
    Path(target).write_text(json.dumps(site, indent=1), encoding="utf-8")
    print(f"{target}: {', '.join(site['datasets'])}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "recommender-results.json")
