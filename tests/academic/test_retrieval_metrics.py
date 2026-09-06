from __future__ import annotations

import math

import pytest

from evals.academic_retrieval.metrics import (
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_recall_at_k_finds_fraction_of_all_relevant_items() -> None:
    result = recall_at_k(
        retrieved_ids=["c3", "c1", "c8", "c5"],
        relevant_ids={"c1", "c5", "c9"},
        k=4,
    )

    assert result == pytest.approx(2.0 / 3.0)


def test_precision_at_k_uses_k_as_denominator() -> None:
    result = precision_at_k(
        retrieved_ids=["c3", "c1", "c8", "c5"],
        relevant_ids={"c1", "c5", "c9"},
        k=4,
    )

    assert result == pytest.approx(0.5)


def test_precision_penalises_unfilled_result_slots() -> None:
    result = precision_at_k(
        retrieved_ids=["c1"],
        relevant_ids={"c1"},
        k=4,
    )

    assert result == pytest.approx(0.25)


def test_reciprocal_rank_uses_first_relevant_result() -> None:
    result = reciprocal_rank(
        retrieved_ids=["c3", "c1", "c5"],
        relevant_ids={"c1", "c5"},
    )

    assert result == pytest.approx(0.5)


def test_reciprocal_rank_is_zero_when_nothing_is_retrieved() -> None:
    result = reciprocal_rank(
        retrieved_ids=[],
        relevant_ids={"c1"},
    )

    assert result == 0.0


def test_duplicate_results_do_not_inflate_metrics() -> None:
    retrieved_ids = ["c1", "c1", "c2"]
    relevant_ids = {"c1", "c2", "c3"}

    assert recall_at_k(
        retrieved_ids,
        relevant_ids,
        k=2,
    ) == pytest.approx(2.0 / 3.0)

    assert precision_at_k(
        retrieved_ids,
        relevant_ids,
        k=2,
    ) == pytest.approx(1.0)


def test_perfect_ndcg_is_one() -> None:
    result = ndcg_at_k(
        retrieved_ids=["c1", "c2", "c3"],
        relevance_by_id={
            "c1": 2.0,
            "c2": 1.0,
            "c3": 0.0,
        },
        k=3,
    )

    assert result == pytest.approx(1.0)


def test_ndcg_rewards_putting_stronger_evidence_first() -> None:
    relevance_by_id = {
        "strong": 2.0,
        "partial": 1.0,
    }

    better = ndcg_at_k(
        ["strong", "partial"],
        relevance_by_id,
        k=2,
    )
    worse = ndcg_at_k(
        ["partial", "strong"],
        relevance_by_id,
        k=2,
    )

    assert better == pytest.approx(1.0)
    assert 0.0 < worse < better


@pytest.mark.parametrize(
    "metric",
    [
        lambda: recall_at_k(["c1"], {"c1"}, 0),
        lambda: precision_at_k(["c1"], {"c1"}, -1),
        lambda: ndcg_at_k(["c1"], {"c1": 1.0}, True),
    ],
)
def test_metrics_reject_invalid_k(metric) -> None:
    with pytest.raises(
        ValueError,
        match="k must be a positive integer",
    ):
        metric()


@pytest.mark.parametrize(
    "metric",
    [
        lambda: recall_at_k(["c1"], set(), 1),
        lambda: reciprocal_rank(["c1"], set()),
    ],
)
def test_metrics_requiring_relevant_items_reject_empty_qrels(
    metric,
) -> None:
    with pytest.raises(
        ValueError,
        match="at least one relevant item",
    ):
        metric()


@pytest.mark.parametrize(
    "invalid_grade",
    [-1.0, math.inf, -math.inf, math.nan],
)
def test_ndcg_rejects_invalid_relevance_grades(
    invalid_grade: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="finite and non-negative",
    ):
        ndcg_at_k(
            ["c1"],
            {"c1": invalid_grade},
            k=1,
        )


def test_ndcg_requires_positive_relevance() -> None:
    with pytest.raises(
        ValueError,
        match="positively relevant item",
    ):
        ndcg_at_k(
            ["c1"],
            {"c1": 0.0},
            k=1,
        )
