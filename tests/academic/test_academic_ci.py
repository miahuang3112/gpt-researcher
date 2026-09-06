from __future__ import annotations

from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = PROJECT_ROOT / ".github/workflows/academic-ci.yml"


def _workflow() -> dict:
    return yaml.load(
        WORKFLOW_PATH.read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )


def test_academic_ci_has_clean_mysql_and_redis_services() -> None:
    workflow = _workflow()
    job = workflow["jobs"]["academic-fast-gate"]
    services = job["services"]

    assert set(services) == {"mysql", "redis"}
    assert services["mysql"]["image"] == "mysql:8.4"
    assert services["redis"]["image"] == "redis:7.4-alpine"
    assert "--health-cmd" in services["mysql"]["options"]
    assert "--health-cmd" in services["redis"]["options"]


def test_academic_ci_runs_migrations_and_fast_tests_without_real_e2e() -> None:
    workflow = _workflow()
    job = workflow["jobs"]["academic-fast-gate"]
    steps = {step["name"]: step for step in job["steps"]}

    migration_command = steps["Verify clean MySQL migration cycle"]["run"]
    assert "upgrade head" in migration_command
    assert "downgrade base" in migration_command
    assert migration_command.count("upgrade head") == 2

    assert steps["Run Academic fast gate"]["run"] == (
        "python -m pytest tests/academic"
    )
    assert job["env"]["RUN_ACADEMIC_E2E"] == "0"
    assert "ollama" not in job["services"]
