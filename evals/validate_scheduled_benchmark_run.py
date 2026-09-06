from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ScheduledBenchmarkError(RuntimeError):
    """Raised when a scheduled full benchmark is incomplete or unsafe."""


def _load(output_dir: Path, name: str) -> dict[str, Any]:
    path = output_dir / name
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ScheduledBenchmarkError(f"Expected JSON object: {path}")
    return payload


def validate_scheduled_run(output_dir: Path) -> dict[str, Any]:
    reports = {
        "ret09": _load(output_dir, "ret09-development.json"),
        "ret10b": _load(output_dir, "ret10b-heldout.json"),
        "ret11a": _load(output_dir, "ret11a-selectors.json"),
        "ret11b": _load(output_dir, "ret11b-selector-quality.json"),
        "data05b": _load(output_dir, "data05b-calibration.json"),
        "eval02a": _load(output_dir, "eval02a-reliability.json"),
        "eval02b": _load(output_dir, "eval02b-validation.json"),
        "eval03": _load(output_dir, "eval03-ablations.json"),
    }
    checks = {
        "ret09_development_only": (
            reports["ret09"]["selection_scope"] == "development-only"
            and reports["ret09"]["test_split_accessed"] is False
        ),
        "ret09_zero_failures": (
            reports["ret09"]["selected_development_metrics"]["failure_count"]
            == 0
        ),
        "ret10b_heldout_selection_forbidden": (
            reports["ret10b"]["evaluation_scope"] == "held-out-test"
            and reports["ret10b"]["parameter_selection_permitted"] is False
        ),
        "ret10b_zero_failures": (
            reports["ret10b"]["default_system_summary"]["failure_count"] == 0
        ),
        "ret11a_zero_failures": all(
            reports["ret11a"]["summary"][system]["failure_count"] == 0
            for system in ("section", "hybrid")
        )
        and reports["ret11a"]["summary"]["hybrid"]["index_build"][
            "failure_count"
        ]
        == 0,
        "ret11b_zero_failures": all(
            reports["ret11b"]["summary"][system]["failure_count"] == 0
            for system in ("section", "hybrid")
        ),
        "data05b_development_only": reports["data05b"]["split"] == "development",
        "data05b_all_cases_predicted": (
            len(reports["data05b"]["predictions"])
            == reports["data05b"]["sample_count"]
        ),
        "eval02a_zero_terminal_failures": (
            reports["eval02a"]["terminal_failure_count"] == 0
        ),
        "eval02b_zero_failures": (
            reports["eval02b"]["citation_validation"]["failure_count"] == 0
            and reports["eval02b"]["contradiction_detection"]["failure_count"]
            == 0
        ),
        "eval02b_heldout_not_used_for_retuning": (
            reports["eval02b"]["interpretation_boundary"][
                "heldout_used_for_retuning"
            ]
            is False
        ),
        "eval03_development_only": (
            reports["eval03"]["selection_scope"] == "development-only"
            and reports["eval03"]["interpretation_boundary"]["heldout_used"]
            is False
        ),
        "eval03_zero_terminal_failures": all(
            summary["terminal_failure_count"] == 0
            for summary in reports["eval03"]["config_summaries"].values()
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    artifacts = {}
    for path in sorted(output_dir.glob("*.json")):
        if path.name == "scheduled-run-manifest.json":
            continue
        artifacts[path.name] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
        }
    manifest = {
        "format_version": 1,
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if not failed else "failed",
        "checks": checks,
        "failed_checks": failed,
        "artifacts": artifacts,
    }
    manifest_path = output_dir / "scheduled-run-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if failed:
        raise ScheduledBenchmarkError(
            "Scheduled benchmark checks failed: " + ", ".join(failed)
        )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate_scheduled_run(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
