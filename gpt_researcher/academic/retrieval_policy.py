from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FrozenRetrievalPolicy:
    policy_id: str
    retriever: str
    enabled_by_default: bool
    embedding_model: str
    candidate_top_k: int
    rrf_k: int
    evidence_top_k_default: int


FROZEN_RETRIEVAL_POLICY = FrozenRetrievalPolicy(
    policy_id="ret10b-hybrid-v1",
    retriever="hybrid",
    enabled_by_default=True,
    embedding_model="qwen3-embedding:4b",
    candidate_top_k=20,
    rrf_k=100,
    evidence_top_k_default=4,
)
