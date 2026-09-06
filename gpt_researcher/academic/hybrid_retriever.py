import asyncio
import logging
from dataclasses import dataclass
from typing import Protocol

from .schemas import EvidenceChunk


logger = logging.getLogger(__name__)


class HybridRetrievalError(RuntimeError):
    """BM25 and vector retrieval both failed."""


class KeywordRetriever(Protocol):
    """
    关键词检索器接口。

    当前实现对应 BM25Retriever。
    """

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[EvidenceChunk]:
        ...


class SemanticRetriever(Protocol):
    """
    语义检索器接口。

    当前实现对应 VectorRetriever。
    """

    async def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[EvidenceChunk]:
        ...


@dataclass
class _FusionRecord:
    """
    RRF融合过程中的内部记录。
    """

    chunk: EvidenceChunk
    fused_score: float = 0.0
    best_rank: int = 1_000_000

    bm25_score: float | None = None
    vector_score: float | None = None


class HybridRetriever:
    """
    使用 Reciprocal Rank Fusion 融合BM25和向量检索结果。
    """

    def __init__(
        self,
        bm25_retriever: KeywordRetriever,
        vector_retriever: SemanticRetriever,
        rrf_k: int = 60,
        max_chunks_per_source: int | None = None,
    ) -> None:
        if rrf_k < 1:
            raise ValueError("rrf_k must be at least 1")
        if max_chunks_per_source is not None and max_chunks_per_source < 1:
            raise ValueError("max_chunks_per_source must be at least 1")

        self.bm25_retriever = bm25_retriever
        self.vector_retriever = vector_retriever
        self.rrf_k = rrf_k
        # Opt-in only: a single-paper evaluation would otherwise discard useful
        # evidence. Enable this after a multi-paper benchmark justifies the cap.
        self.max_chunks_per_source = max_chunks_per_source

    async def search(
        self,
        query: str,
        top_k: int = 5,
        candidate_top_k: int | None = None,
    ) -> list[EvidenceChunk]:
        """
        并行执行BM25和向量检索，然后通过RRF融合排名。

        candidate_top_k控制每一路召回多少候选证据。
        top_k控制最终返回多少条证据。
        """
        if not query.strip():
            raise ValueError("query must not be blank")

        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        if candidate_top_k is None:
            candidate_top_k = max(top_k * 2, 10)

        if candidate_top_k < top_k:
            raise ValueError(
                "candidate_top_k must be greater than "
                "or equal to top_k"
            )

        bm25_outcome, vector_outcome = await asyncio.gather(
            asyncio.to_thread(
                self.bm25_retriever.search,
                query,
                top_k=candidate_top_k,
            ),
            self.vector_retriever.search(
                query,
                top_k=candidate_top_k,
            ),
            return_exceptions=True,
        )

        bm25_error = (
            bm25_outcome
            if isinstance(bm25_outcome, Exception)
            else None
        )
        vector_error = (
            vector_outcome
            if isinstance(vector_outcome, Exception)
            else None
        )

        if bm25_error is not None and vector_error is not None:
            raise HybridRetrievalError(
                "BM25 and vector retrieval both failed. "
                f"BM25: {type(bm25_error).__name__}: {bm25_error}; "
                "Vector: "
                f"{type(vector_error).__name__}: {vector_error}"
            ) from vector_error

        if bm25_error is not None:
            logger.warning(
                "BM25 retrieval failed; using vector results only: %s",
                bm25_error,
            )

        if vector_error is not None:
            logger.warning(
                "Vector retrieval failed; using BM25 results only: %s",
                vector_error,
            )

        bm25_results = (
            [] if bm25_error is not None else bm25_outcome
        )
        vector_results = (
            [] if vector_error is not None else vector_outcome
        )

        return self._rrf_fuse(
            bm25_results=bm25_results,
            vector_results=vector_results,
            top_k=top_k,
        )

    def _rrf_fuse(
        self,
        bm25_results: list[EvidenceChunk],
        vector_results: list[EvidenceChunk],
        top_k: int,
    ) -> list[EvidenceChunk]:
        records: dict[str, _FusionRecord] = {}

        self._add_ranking(
            records=records,
            results=bm25_results,
            source="bm25",
        )

        self._add_ranking(
            records=records,
            results=vector_results,
            source="vector",
        )

        ranked_records = sorted(
            records.values(),
            key=lambda record: (
                -record.fused_score,
                record.best_rank,
                record.chunk.chunk_id,
            ),
        )

        fused_results: list[EvidenceChunk] = []

        source_counts: dict[str, int] = {}

        for record in ranked_records:
            source_id = record.chunk.paper_id
            if (
                self.max_chunks_per_source is not None
                and source_counts.get(source_id, 0) >= self.max_chunks_per_source
            ):
                continue
            fused_results.append(
                record.chunk.model_copy(
                    update={
                        "bm25_score": record.bm25_score,
                        "vector_score": record.vector_score,
                        "fused_score": record.fused_score,
                    }
                )
            )
            source_counts[source_id] = source_counts.get(source_id, 0) + 1
            if len(fused_results) >= top_k:
                break

        return fused_results

    def _add_ranking(
        self,
        records: dict[str, _FusionRecord],
        results: list[EvidenceChunk],
        source: str,
    ) -> None:
        """
        把一条检索排名加入RRF记录。

        每个检索器中的重复chunk只计算一次。
        """
        seen_chunk_ids: set[str] = set()

        for rank, chunk in enumerate(results, start=1):
            if chunk.chunk_id in seen_chunk_ids:
                continue

            seen_chunk_ids.add(chunk.chunk_id)

            record = records.get(chunk.chunk_id)

            if record is None:
                record = _FusionRecord(chunk=chunk)
                records[chunk.chunk_id] = record
            else:
                self._validate_matching_chunks(
                    existing=record.chunk,
                    incoming=chunk,
                )

            record.fused_score += 1.0 / (
                self.rrf_k + rank
            )
            record.best_rank = min(
                record.best_rank,
                rank,
            )

            if source == "bm25":
                record.bm25_score = chunk.bm25_score
            elif source == "vector":
                record.vector_score = chunk.vector_score
            else:
                raise ValueError(
                    f"Unsupported ranking source: {source}"
                )

    @staticmethod
    def _validate_matching_chunks(
        existing: EvidenceChunk,
        incoming: EvidenceChunk,
    ) -> None:
        """
        防止两个不同证据错误地使用同一个chunk_id。
        """
        if (
            existing.paper_id != incoming.paper_id
            or existing.text != incoming.text
        ):
            raise RuntimeError(
                "Conflicting evidence chunks share the same chunk_id: "
                f"{existing.chunk_id}"
            )
