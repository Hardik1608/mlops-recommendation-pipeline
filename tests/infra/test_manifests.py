"""Static checks of deployment manifests (no Docker / Kubernetes / Helm binaries needed).

Beyond "the YAML parses", these assert the properties the design relies on, and that
dashboards/alerts only reference metrics the code actually exports.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_yaml(relative: str):
    return yaml.safe_load((ROOT / relative).read_text(encoding="utf-8"))


# ------------------------------------------------------------------------- compose
@pytest.fixture(scope="module")
def compose() -> dict:
    return load_yaml("docker-compose.yml")


def test_compose_has_the_full_stack(compose) -> None:
    services = set(compose["services"])
    assert {
        "kafka",
        "kafka-init",
        "redis",
        "mlflow",
        "trainer",
        "api",
        "consumer",
        "producer",
        "prometheus",
        "pushgateway",
        "grafana",
        "airflow",
    } <= services
    assert "zookeeper" not in services


def test_kafka_runs_in_kraft_mode(compose) -> None:
    env = compose["services"]["kafka"]["environment"]
    assert "controller" in env["KAFKA_PROCESS_ROLES"]
    assert env["KAFKA_AUTO_CREATE_TOPICS_ENABLE"] == "false"  # topics are created explicitly


def test_compose_images_are_pinned(compose) -> None:
    for name, service in compose["services"].items():
        image = service.get("image", "")
        assert not image.endswith(":latest"), name
        assert ":" in image, f"{name} has an unpinned image"


def test_api_waits_for_a_registered_model(compose) -> None:
    api = compose["services"]["api"]
    assert api["depends_on"]["trainer"]["condition"] == "service_completed_successfully"
    assert api["environment"]["RECSYS_MODEL_SOURCE"] == "mlflow"
    assert compose["services"]["airflow"]["profiles"] == ["airflow"]


# ------------------------------------------------------------------------- helm
def test_helm_chart_metadata_and_values() -> None:
    chart = load_yaml("deploy/helm/recsys/Chart.yaml")
    assert chart["apiVersion"] == "v2" and chart["name"] == "recsys"
    values = load_yaml("deploy/helm/recsys/values.yaml")
    api = values["api"]
    assert api["livenessProbe"]["httpGet"]["path"] == "/health"
    assert api["readinessProbe"]["httpGet"]["path"] == "/ready"
    assert api["autoscaling"]["enabled"] is True
    assert api["autoscaling"]["minReplicas"] >= 2
    assert api["resources"]["requests"] and api["resources"]["limits"]
    assert values["securityContext"]["readOnlyRootFilesystem"] is True
    assert "FEAST_REGISTRY_URL" not in values["config"]  # secrets never live in values


def test_helm_templates_exist() -> None:
    templates = {p.name for p in (ROOT / "deploy/helm/recsys/templates").iterdir()}
    assert {
        "deployment-api.yaml",
        "deployment-consumer.yaml",
        "service.yaml",
        "hpa.yaml",
        "configmap.yaml",
        "pdb.yaml",
        "_helpers.tpl",
    } <= templates


# ------------------------------------------------------------------------- monitoring
def _exported_metric_names() -> set[str]:
    from prometheus_client import Counter, Gauge, Histogram

    import recsys.serving.metrics as serving_metrics

    names = set()
    for obj in vars(serving_metrics).values():
        if isinstance(obj, Counter | Gauge | Histogram):
            names.add(obj._name)
    drift_source = (ROOT / "src/recsys/monitoring/drift.py").read_text(encoding="utf-8")
    names |= set(re.findall(r'Gauge\(\s*"(recsys_[a-z_]+)"', drift_source))
    return names


def _referenced_metrics(text: str) -> set[str]:
    found = set(re.findall(r"\b(recsys_[a-z_]+)", text))
    stripped = set()
    for name in found:
        for suffix in ("_bucket", "_count", "_sum", "_total"):
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                break
        stripped.add(name)
    return stripped


def test_alerts_and_dashboard_only_use_exported_metrics() -> None:
    exported = _exported_metric_names()
    exported_base = {n.removesuffix("_total") for n in exported}
    alerts = (ROOT / "deploy/prometheus/alerts.yml").read_text(encoding="utf-8")
    dashboard = (ROOT / "deploy/grafana/dashboards/recsys.json").read_text(encoding="utf-8")
    for source in (alerts, dashboard):
        missing = _referenced_metrics(source) - exported_base
        assert not missing, f"unknown metrics referenced: {missing}"


def test_prometheus_scrapes_the_api_and_pushgateway() -> None:
    config = load_yaml("deploy/prometheus/prometheus.yml")
    jobs = {job["job_name"]: job for job in config["scrape_configs"]}
    assert jobs["recsys-api"]["static_configs"][0]["targets"] == ["api:8000"]
    assert jobs["pushgateway"]["honor_labels"] is True


def test_latency_slo_alert_is_100ms() -> None:
    rules = {
        rule.get("alert") or rule.get("record"): rule
        for group in load_yaml("deploy/prometheus/alerts.yml")["groups"]
        for rule in group["rules"]
    }
    assert rules["RecsysHighP95Latency"]["expr"].strip().endswith("> 0.1")
    assert {"RecsysHighErrorRate", "RecsysApiDown", "RecsysFeatureDrift"} <= set(rules)


def test_grafana_dashboard_uses_the_provisioned_datasource() -> None:
    datasource = load_yaml("deploy/grafana/provisioning/datasources/prometheus.yml")
    uid = datasource["datasources"][0]["uid"]
    dashboard = json.loads((ROOT / "deploy/grafana/dashboards/recsys.json").read_text("utf-8"))
    panels = [p for p in dashboard["panels"] if p["type"] != "row"]
    assert len(panels) >= 10
    for panel in panels:
        assert panel["datasource"]["uid"] == uid
        assert panel["targets"], panel["title"]


# ------------------------------------------------------------------------- ci / feast
def test_ci_workflow_jobs() -> None:
    workflow = load_yaml(".github/workflows/ci.yml")
    assert {"lint", "test", "docker", "helm", "airflow-dags", "monitoring", "e2e-smoke"} <= set(
        workflow["jobs"]
    )


@pytest.mark.parametrize(
    "name", ["feature_store.yaml", "feature_store.redis.yaml", "feature_store.k8s.yaml"]
)
def test_feast_configs_parse(name) -> None:
    config = load_yaml(f"feature_repo/{name}")
    assert config["project"] == "recsys"
    assert config["entity_key_serialization_version"] == 3
