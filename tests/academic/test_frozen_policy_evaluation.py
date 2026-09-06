from __future__ import annotations

import pytest

from evals.academic_retrieval.evaluate_frozen_policy import validate_frozen_policy


def _policy() -> dict:
    return {
        "status": "frozen",
        "selection_scope": "development-only",
        "retriever": "hybrid",
        "parameters": {
            "embedding_model": "embedding-test",
            "candidate_top_k": 20,
            "rrf_k": 100,
            "evaluation_cutoffs": [4, 10],
        },
        "integrity": {"retuning_after_test_permitted": False},
    }


def test_frozen_policy_is_accepted_without_parameter_overrides() -> None:
    parameters = validate_frozen_policy(_policy())
    assert parameters == _policy()["parameters"]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"status": "draft"}, "frozen retrieval policy"),
        ({"selection_scope": "test"}, "development only"),
        ({"retriever": "bm25"}, "frozen Hybrid"),
        ({"integrity": {"retuning_after_test_permitted": True}}, "forbid"),
    ],
)
def test_held_out_evaluation_rejects_unfrozen_or_tunable_policy(
    mutation,
    message,
) -> None:
    policy = _policy()
    policy.update(mutation)
    with pytest.raises(ValueError, match=message):
        validate_frozen_policy(policy)
