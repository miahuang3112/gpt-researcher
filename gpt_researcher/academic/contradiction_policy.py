from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FrozenContradictionPolicy:
    policy_id: str
    model: str
    warning_confidence_threshold: float
    positive_relations: frozenset[str]


FROZEN_CONTRADICTION_POLICY = FrozenContradictionPolicy(
    policy_id="data05b-qwen35-9b-chatgpt-dev-v1",
    model="qwen3.5:9b",
    warning_confidence_threshold=0.95,
    positive_relations=frozenset({
        "contradiction",
        "partial_contradiction",
    }),
)
