import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_PATH = ROOT / "docs/academic_resume_evidence.v1.json"
BULLETS_PATH = ROOT / "docs/academic_resume_bullets.md"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_resume_evidence_sources_are_hash_pinned() -> None:
    evidence = _load(EVIDENCE_PATH)

    for source in evidence["sources"].values():
        path = ROOT / source["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == source["sha256"]


def test_resume_quality_metrics_match_versioned_summaries() -> None:
    evidence = _load(EVIDENCE_PATH)
    facts = evidence["facts"]
    eval02a = _load(ROOT / evidence["sources"]["eval02a"]["path"])
    eval02b = _load(ROOT / evidence["sources"]["eval02b"]["path"])
    eval03 = _load(ROOT / evidence["sources"]["eval03"]["path"])

    assert facts["score_reliability"]["chatgpt_reference_mae"] == (
        eval02a["aggregate"]["chatgpt_reference_mae"]
    )
    assert facts["score_reliability"]["scorable_count"] == eval02a[
        "scorable_count"
    ]
    assert facts["validation_accuracy"]["citation_four_class_accuracy"] == (
        eval02b["citation_validation"]["four_class_accuracy"]
    )
    assert facts["validation_accuracy"]["contradiction_six_class_accuracy"] == (
        eval02b["contradiction_detection"]["six_class_accuracy"]
    )
    assert facts["development_ablation"]["hybrid_anchored_mae"] == (
        eval03["configuration_metrics"]["hybrid_anchored"][
            "chatgpt_reference_mae"
        ]
    )
    assert facts["development_ablation"]["heldout_used"] is False


def test_resume_bullets_preserve_ai_reference_and_decision_boundaries() -> None:
    bullets = BULLETS_PATH.read_text(encoding="utf-8")

    assert "ChatGPT 参考" in bullets
    assert "ChatGPT-reference" in bullets
    assert "not human gold" in bullets
    assert "拒绝使用测试集调参" in bullets
    assert "without test-set retuning" in bullets
    assert "not peer review" in bullets
