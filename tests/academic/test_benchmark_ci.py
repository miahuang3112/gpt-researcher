import shutil
import subprocess
from pathlib import Path

import pytest

from evals import verify_academic_benchmarks as verifier
from evals.validate_scheduled_benchmark_run import validate_scheduled_run


ROOT = Path(__file__).resolve().parents[2]


def test_deterministic_benchmark_smoke_passes_versioned_evidence() -> None:
    report = verifier.verify_versioned_benchmarks()

    assert report["status"] == "passed"
    assert len(report["hash_checks"]) == 11
    assert all(report["invariants"].values())


def test_deterministic_benchmark_smoke_fails_closed_on_hash_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(verifier, "_sha256", lambda _path: "0" * 64)

    with pytest.raises(verifier.BenchmarkVerificationError):
        verifier.verify_versioned_benchmarks()


def test_scheduled_run_validator_accepts_complete_zero_failure_reports(
    tmp_path: Path,
) -> None:
    sources = {
        "ret09-development.json": ROOT / "evals/academic_retrieval/results/ret09-development-tuning-qwen3-embedding-4b.json",
        "ret10b-heldout.json": ROOT / "evals/academic_retrieval/results/ret10b-heldout-test-qwen3-embedding-4b.json",
        "ret11a-selectors.json": ROOT / "evals/academic_retrieval/results/ret11a-section-vs-hybrid-heldout.json",
        "ret11b-selector-quality.json": ROOT / "evals/academic_quality/results/ret11b-section-vs-hybrid-heldout.json",
        "data05b-calibration.json": ROOT / "evals/academic_quality/results/data05b-contradiction-calibration.json",
        "eval02a-reliability.json": ROOT / "evals/academic_quality/results/eval02a-score-reliability.json",
        "eval02b-validation.json": ROOT / "evals/academic_quality/results/eval02b-validation-accuracy.json",
        "eval03-ablations.json": ROOT / "evals/academic_quality/results/eval03-development-ablations.json",
    }
    for target, source in sources.items():
        shutil.copyfile(source, tmp_path / target)

    manifest = validate_scheduled_run(tmp_path)

    assert manifest["status"] == "passed"
    assert all(manifest["checks"].values())
    assert (tmp_path / "scheduled-run-manifest.json").is_file()


def test_benchmark_workflow_and_runner_are_wired() -> None:
    workflow = (ROOT / ".github/workflows/academic-benchmarks.yml").read_text(
        encoding="utf-8"
    )
    ci = (ROOT / ".github/workflows/academic-ci.yml").read_text(encoding="utf-8")
    runner = ROOT / "scripts/run_academic_benchmarks.sh"

    assert 'cron: "17 3 * * 1"' in workflow
    assert "deterministic-smoke:" in workflow
    assert "full-model-benchmark:" in workflow
    assert "ollama/ollama:0.32.13" in workflow
    assert "actions/upload-artifact@v4" in workflow
    assert "python -m evals.verify_academic_benchmarks" in ci
    assert "scripts/run_academic_benchmarks.sh" in ci
    subprocess.run(["bash", "-n", str(runner)], check=True)
