from __future__ import annotations

import json
from pathlib import Path

import yaml

from backend.academic_api.metrics import AcademicMetricsSnapshot, render_prometheus


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_prometheus_renderer_exposes_required_operational_signals() -> None:
    text = render_prometheus(
        AcademicMetricsSnapshot(
            jobs_by_status={"queued": 3, "running": 1, "failed": 2},
            duration_seconds=(10.0, 20.0),
            retry_events=4,
            model_failures=2,
            queue_stream_length=8,
            queue_pending=3,
            queue_dlq_length=1,
            mysql_up=True,
            redis_up=False,
        )
    )

    assert 'academic_jobs{status="queued"} 3' in text
    assert "academic_queue_pending 3" in text
    assert "academic_queue_dlq_length 1" in text
    assert "academic_retry_events_total 4" in text
    assert "academic_model_failures_total 2" in text
    assert "academic_evaluation_duration_seconds_count 2" in text
    assert "academic_evaluation_duration_seconds_sum 30.0" in text
    assert "academic_evaluation_duration_seconds_avg 15.0" in text
    assert "academic_metrics_mysql_up 1" in text
    assert "academic_metrics_redis_up 0" in text
    assert text.endswith("\n")


def test_observability_profile_wires_prometheus_and_grafana() -> None:
    compose = yaml.safe_load(
        (PROJECT_ROOT / "compose.academic.yml").read_text(encoding="utf-8")
    )
    services = compose["services"]

    assert services["prometheus"]["profiles"] == ["observability"]
    assert services["grafana"]["profiles"] == ["observability"]
    assert services["prometheus"]["depends_on"]["academic-api"]["condition"] == (
        "service_healthy"
    )
    assert services["grafana"]["depends_on"]["prometheus"]["condition"] == (
        "service_healthy"
    )


def test_alerts_and_dashboard_cover_stretch_requirements() -> None:
    alerts = yaml.safe_load(
        (PROJECT_ROOT / "observability/prometheus/alerts.yml").read_text(
            encoding="utf-8"
        )
    )
    expressions = "\n".join(
        rule["expr"] for group in alerts["groups"] for rule in group["rules"]
    )
    for metric in (
        "academic_queue_pending",
        "academic_queue_dlq_length",
        "academic_model_failures_total",
        "academic_evaluation_duration_seconds_avg",
    ):
        assert metric in expressions

    dashboard = json.loads(
        (PROJECT_ROOT / "observability/grafana/dashboards/academic-overview.json").read_text(
            encoding="utf-8"
        )
    )
    dashboard_expressions = "\n".join(
        target["expr"]
        for panel in dashboard["panels"]
        for target in panel["targets"]
    )
    for metric in (
        "academic_jobs",
        "academic_queue_pending",
        "academic_queue_dlq_length",
        "academic_retry_events_total",
        "academic_model_failures_total",
        "academic_evaluation_duration_seconds_avg",
    ):
        assert metric in dashboard_expressions
