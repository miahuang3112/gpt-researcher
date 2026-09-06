from __future__ import annotations

import math
from collections.abc import Collection, Mapping, Sequence


def _validate_k(k: int) -> None:
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")


def _unique_top_k(
    retrieved_ids: Sequence[str],
    k: int,
) -> list[str]:
    """Return the first K unique result IDs while preserving rank."""

    unique_ids: list[str] = []
    seen_ids: set[str] = set()

    for item_id in retrieved_ids:
        if item_id in seen_ids:
            continue

        seen_ids.add(item_id)
        unique_ids.append(item_id)

        if len(unique_ids) == k:
            break

    return unique_ids


def _require_relevant_items(
    relevant_ids: Collection[str],
) -> set[str]:
    relevant_set = set(relevant_ids)

    if not relevant_set:
        raise ValueError(
            "at least one relevant item is required"
        )

    return relevant_set


def recall_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Collection[str],
    k: int,
) -> float:
    """Return the fraction of all relevant items found in the Top-K."""

    _validate_k(k)
    relevant_set = _require_relevant_items(relevant_ids)
    top_k = _unique_top_k(retrieved_ids, k)

    retrieved_relevant_count = sum(
        item_id in relevant_set
        for item_id in top_k
    )

    return retrieved_relevant_count / len(relevant_set)


def precision_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Collection[str],
    k: int,
) -> float:
    """Return the fraction of the K result slots that are relevant."""

    _validate_k(k)
    relevant_set = set(relevant_ids)
    top_k = _unique_top_k(retrieved_ids, k)

    retrieved_relevant_count = sum(
        item_id in relevant_set
        for item_id in top_k
    )

    return retrieved_relevant_count / k


def reciprocal_rank(
    retrieved_ids: Sequence[str],
    relevant_ids: Collection[str],
) -> float:
    """Return the reciprocal rank of the first relevant result."""

    relevant_set = _require_relevant_items(relevant_ids)
    ranked_ids = _unique_top_k(
        retrieved_ids,
        len(retrieved_ids),
    )

    for rank, item_id in enumerate(
        ranked_ids,
        start=1,
    ):
        if item_id in relevant_set:
            return 1.0 / rank

    return 0.0


def ndcg_at_k(
    retrieved_ids: Sequence[str],
    relevance_by_id: Mapping[str, float],
    k: int,
) -> float:
    """
    Return Normalised Discounted Cumulative Gain at K.

    Relevance grades must be finite and non-negative. A common academic
    evidence scale is 0=irrelevant, 1=partially relevant, 2=highly relevant.
    """

    _validate_k(k)

    relevance_values = list(relevance_by_id.values())

    if any(
        not math.isfinite(value) or value < 0
        for value in relevance_values
    ):
        raise ValueError(
            "relevance grades must be finite and non-negative"
        )

    positive_relevances = [
        value
        for value in relevance_values
        if value > 0
    ]

    if not positive_relevances:
        raise ValueError(
            "at least one positively relevant item is required"
        )

    def discounted_gain(
        relevance_values_by_rank: Sequence[float],
    ) -> float:
        return sum(
            (2.0**relevance - 1.0)
            / math.log2(rank + 1)
            for rank, relevance in enumerate(
                relevance_values_by_rank,
                start=1,
            )
        )

    top_k = _unique_top_k(retrieved_ids, k)
    actual_relevances = [
        relevance_by_id.get(item_id, 0.0)
        for item_id in top_k
    ]

    ideal_relevances = sorted(
        positive_relevances,
        reverse=True,
    )[:k]

    dcg = discounted_gain(actual_relevances)
    ideal_dcg = discounted_gain(ideal_relevances)

    return dcg / ideal_dcg
