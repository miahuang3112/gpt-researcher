from __future__ import annotations

from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _services() -> dict:
    compose_path = PROJECT_ROOT / "compose.academic.yml"
    return yaml.safe_load(compose_path.read_text(encoding="utf-8"))["services"]


def test_migration_is_a_one_shot_alembic_service() -> None:
    services = _services()
    migration = services["academic-migrate"]

    assert migration["restart"] == "no"
    assert migration["depends_on"]["mysql"]["condition"] == "service_healthy"
    assert migration["command"] == [
        "python",
        "-m",
        "alembic",
        "-c",
        "alembic.ini",
        "upgrade",
        "head",
    ]


def test_api_and_worker_are_blocked_until_migration_succeeds() -> None:
    services = _services()

    for service_name in ("academic-api", "academic-worker"):
        migration_dependency = services[service_name]["depends_on"][
            "academic-migrate"
        ]
        assert migration_dependency["condition"] == "service_completed_successfully"


def test_api_and_worker_have_dependency_aware_readiness_checks() -> None:
    services = _services()
    api_healthcheck = services["academic-api"]["healthcheck"]["test"]
    worker_healthcheck = services["academic-worker"]["healthcheck"]["test"]

    assert "/health/ready" in api_healthcheck[-1]
    assert worker_healthcheck == [
        "CMD",
        "python",
        "-m",
        "backend.academic_api.health",
        "worker-ready",
    ]


def test_worker_has_opt_in_external_scholarly_evidence_configuration() -> None:
    environment = _services()["academic-worker"]["environment"]

    assert environment["ACADEMIC_EXTERNAL_EVIDENCE_ENABLED"].endswith(
        ":-false}"
    )
    assert environment["ACADEMIC_EXTERNAL_EVIDENCE_LIMIT"].endswith(":-5}")
    assert "OPENALEX_API_KEY" in environment
    assert "CROSSREF_MAILTO" in environment
