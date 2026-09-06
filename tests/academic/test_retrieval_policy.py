from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from backend.academic_api.config import Settings
from gpt_researcher.academic.retrieval_policy import FROZEN_RETRIEVAL_POLICY


ROOT = Path(__file__).resolve().parents[2]
POLICY_PATH = ROOT / "evals/academic_retrieval/ret10b_frozen_policy.json"


def test_frozen_policy_matches_production_defaults() -> None:
    payload = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    parameters = payload["parameters"]
    policy = FROZEN_RETRIEVAL_POLICY

    assert payload["policy_id"] == policy.policy_id
    assert payload["retriever"] == policy.retriever
    assert payload["enabled_by_default"] == policy.enabled_by_default
    assert parameters["embedding_model"] == policy.embedding_model
    assert parameters["candidate_top_k"] == policy.candidate_top_k
    assert parameters["rrf_k"] == policy.rrf_k
    assert parameters["evidence_top_k_default"] == policy.evidence_top_k_default

    fields = Settings.model_fields
    assert fields["academic_hybrid_retrieval_enabled"].default is True
    assert fields["academic_hybrid_candidate_top_k"].default == policy.candidate_top_k
    assert fields["academic_hybrid_rrf_k"].default == policy.rrf_k
    assert fields["academic_embedding_model"].default == policy.embedding_model


def test_deployment_defaults_match_frozen_policy() -> None:
    compose = yaml.safe_load((ROOT / "compose.academic.yml").read_text(encoding="utf-8"))
    expected = {
        "ACADEMIC_HYBRID_RETRIEVAL_ENABLED": "${ACADEMIC_HYBRID_RETRIEVAL_ENABLED:-true}",
        "ACADEMIC_HYBRID_CANDIDATE_TOP_K": "${ACADEMIC_HYBRID_CANDIDATE_TOP_K:-20}",
        "ACADEMIC_HYBRID_RRF_K": "${ACADEMIC_HYBRID_RRF_K:-100}",
        "ACADEMIC_EMBEDDING_MODEL": "${ACADEMIC_EMBEDDING_MODEL:-qwen3-embedding:4b}",
    }
    for service_name in ("academic-api", "academic-worker"):
        environment = compose["services"][service_name]["environment"]
        assert {key: environment[key] for key in expected} == expected


def test_frozen_policy_detects_query_mapping_drift() -> None:
    payload = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    query_mapping = ROOT / "gpt_researcher/academic/retrieval_queries.py"
    observed_hash = hashlib.sha256(query_mapping.read_bytes()).hexdigest()
    assert observed_hash == payload["development_evidence"]["query_mapping_sha256"]


def test_frozen_policy_forbids_test_driven_retuning() -> None:
    payload = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    assert payload["status"] == "frozen"
    assert payload["selection_scope"] == "development-only"
    assert payload["integrity"]["held_out_test_used_for_selection"] is False
    assert payload["integrity"]["retuning_after_test_permitted"] is False
