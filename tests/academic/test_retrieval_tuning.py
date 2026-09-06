from __future__ import annotations

import pytest

from evals.academic_retrieval.tune_retrieval import (
    CachedEmbeddingClient,
    assert_complete_trial,
    select_best_trial,
)


class CountingEmbeddingClient:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def embed(self, texts):
        values = list(texts)
        self.calls.append(values)
        return [[float(len(text))] for text in values]


@pytest.mark.asyncio
async def test_cached_embedding_client_only_embeds_missing_texts() -> None:
    base = CountingEmbeddingClient()
    cached = CachedEmbeddingClient(base)

    assert await cached.embed(["one", "two"]) == [[3.0], [3.0]]
    assert await cached.embed(["two", "three", "one"]) == [[3.0], [5.0], [3.0]]
    assert base.calls == [["one", "two"], ["three"]]


def _trial(recall: float, ndcg: float, candidate_top_k: int, rrf_k: int):
    return {
        "config": {"candidate_top_k": candidate_top_k, "rrf_k": rrf_k},
        "summary": {
            "hybrid": {
                "recall_at_k": {"10": recall},
                "ndcg_at_k": {"10": ndcg},
            }
        },
    }


def test_select_best_trial_uses_declared_tie_breakers() -> None:
    trials = [
        _trial(0.8, 0.6, 40, 60),
        _trial(0.8, 0.7, 40, 60),
        _trial(0.8, 0.7, 20, 100),
        _trial(0.8, 0.7, 20, 20),
    ]

    selected = select_best_trial(trials, cutoff=10)

    assert selected["config"] == {"candidate_top_k": 20, "rrf_k": 20}


def test_incomplete_trial_cannot_be_used_for_tuning() -> None:
    trial = _trial(0.8, 0.7, 20, 60)
    trial["summary"]["vector"] = {"failure_count": 1}
    trial["summary"]["hybrid"]["failure_count"] = 0

    with pytest.raises(RuntimeError, match="incomplete retrieval trial"):
        assert_complete_trial(trial)
