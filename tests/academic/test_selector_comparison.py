from __future__ import annotations

import hashlib
import json

import pytest

from evals.academic_retrieval.compare_selectors import compare_selectors


class ConstantEmbeddingClient:
    async def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.asyncio
async def test_selector_comparison_reports_evidence_proxy_latency_and_failures(
    tmp_path,
) -> None:
    split_dir = tmp_path / "test"
    split_dir.mkdir()
    corpus_path = split_dir / "corpus.jsonl"
    queries_path = split_dir / "queries.jsonl"
    corpus_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "chunk_id": "c1",
                        "paper_id": "paper-1",
                        "text": "The research problem and objective are explicit. " * 10,
                        "section": "Abstract",
                    }
                ),
                json.dumps(
                    {
                        "chunk_id": "c2",
                        "paper_id": "paper-1",
                        "text": "Implementation details are provided. " * 10,
                        "section": "Methods",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    queries_path.write_text(
        json.dumps(
            {
                "query_id": "q1",
                "paper_id": "paper-1",
                "query": "Where is the research problem and objective?",
                "dimension": "research_question_clarity",
                "relevance_by_chunk_id": {"c1": 2, "c2": 0},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_path = split_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "split_role": "test",
                "tuning_permitted": False,
                "paper_ids": ["paper-1"],
                "corpus_sha256": _sha256(corpus_path),
                "queries_sha256": _sha256(queries_path),
            }
        ),
        encoding="utf-8",
    )
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(
        json.dumps(
            {
                "policy_id": "test-policy",
                "status": "frozen",
                "selection_scope": "development-only",
                "retriever": "hybrid",
                "parameters": {
                    "embedding_model": "test-embedding",
                    "candidate_top_k": 1,
                    "rrf_k": 100,
                    "evidence_top_k_default": 1,
                    "evaluation_cutoffs": [1],
                },
                "integrity": {"retuning_after_test_permitted": False},
            }
        ),
        encoding="utf-8",
    )

    report = await compare_selectors(
        policy_path=policy_path,
        test_manifest_path=manifest_path,
        embedding_client=ConstantEmbeddingClient(),
    )

    assert report["comparison_scope"] == "held-out-test-descriptive-only"
    assert report["policy_changed"] is False
    assert report["summary"]["section"]["failure_count"] == 0
    assert report["summary"]["hybrid"]["failure_count"] == 0
    assert report["summary"]["hybrid"]["index_build"]["failure_count"] == 0
    assert report["summary"]["section"]["evidence"]["recall_at_k"] == 1.0
    assert report["summary"]["hybrid"]["evidence"]["recall_at_k"] == 1.0
    assert report["summary"]["section"]["citation_readiness_proxy"][
        "valid_candidate_rate"
    ] == 1.0
    assert report["interpretation_boundary"][
        "measures_actual_model_citation_choice"
    ] is False
