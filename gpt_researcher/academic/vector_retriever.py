import os
from collections.abc import Sequence
from typing import Protocol

import httpx
import numpy as np
from dotenv import load_dotenv

from .schemas import EvidenceChunk


load_dotenv()


class EmbeddingClient(Protocol):
    """
    Embedding 客户端统一接口。

    VectorRetriever 只依赖这个接口，不直接依赖 Ollama，
    因此测试时可以传入假的 Embedding 客户端。
    """

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        ...


class OllamaEmbeddingClient:
    """
    通过 Ollama /api/embed 接口生成文本向量。
    """

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        timeout: float = 120.0,
        batch_size: int = 32,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        self.model = model or os.getenv(
            "ACADEMIC_EMBEDDING_MODEL",
            "qwen3-embedding:0.6b",
        )
        self.base_url = (
            base_url
            or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        ).rstrip("/")
        self.timeout = timeout
        self.batch_size = batch_size

    async def embed(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        clean_texts = list(texts)

        if not clean_texts:
            raise ValueError("texts must not be empty")

        if any(not text.strip() for text in clean_texts):
            raise ValueError("texts must not contain blank strings")

        all_embeddings: list[list[float]] = []

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for start in range(0, len(clean_texts), self.batch_size):
                batch = clean_texts[start : start + self.batch_size]

                try:
                    response = await client.post(
                        f"{self.base_url}/api/embed",
                        json={
                            "model": self.model,
                            "input": batch,
                        },
                    )
                except httpx.RequestError as exc:
                    raise RuntimeError(
                        f"Could not connect to Ollama at {self.base_url}"
                    ) from exc

                if response.is_error:
                    raise RuntimeError(
                        "Ollama embedding request failed "
                        f"with HTTP {response.status_code}"
                    )

                try:
                    payload = response.json()
                except ValueError as exc:
                    raise RuntimeError(
                        "Ollama returned an invalid JSON response"
                    ) from exc

                embeddings = payload.get("embeddings")

                if not isinstance(embeddings, list):
                    raise RuntimeError(
                        "Ollama response does not contain embeddings"
                    )

                if len(embeddings) != len(batch):
                    raise RuntimeError(
                        "Ollama returned an unexpected number of embeddings"
                    )

                all_embeddings.extend(embeddings)

        return all_embeddings


class VectorRetriever:
    """
    使用 Embedding 和余弦相似度对证据片段进行语义检索。
    """

    def __init__(
        self,
        chunks: Sequence[EvidenceChunk],
        embedding_client: EmbeddingClient | None = None,
    ) -> None:
        self.chunks = list(chunks)

        if not self.chunks:
            raise ValueError("chunks must not be empty")

        if any(not chunk.text.strip() for chunk in self.chunks):
            raise ValueError("chunk text must not be blank")

        self.embedding_client = (
            embedding_client or OllamaEmbeddingClient()
        )

        self._chunk_matrix: np.ndarray | None = None

    async def build_index(self) -> None:
        """
        为全部证据片段生成向量并缓存。

        同一个 VectorRetriever 多次搜索时，不会重复生成文档向量。
        """
        embeddings = await self.embedding_client.embed(
            [chunk.text for chunk in self.chunks]
        )

        self._chunk_matrix = self._normalise_embeddings(
            embeddings,
            expected_rows=len(self.chunks),
        )

    async def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[EvidenceChunk]:
        """
        返回与查询语义最接近的 Top-K 证据片段。
        """
        if not query.strip():
            raise ValueError("query must not be blank")

        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        if self._chunk_matrix is None:
            await self.build_index()

        query_embeddings = await self.embedding_client.embed([query])

        query_matrix = self._normalise_embeddings(
            query_embeddings,
            expected_rows=1,
        )

        if self._chunk_matrix is None:
            raise RuntimeError("vector index was not built")

        if self._chunk_matrix.shape[1] != query_matrix.shape[1]:
            raise RuntimeError(
                "Query embedding dimension does not match "
                "the document embedding dimension"
            )

        # 向量已经归一化，因此点积就是余弦相似度。
        scores = self._chunk_matrix @ query_matrix[0]

        result_count = min(top_k, len(self.chunks))

        # stable 排序保证相同分数时仍保持原始顺序。
        ranked_indices = np.argsort(
            -scores,
            kind="stable",
        )[:result_count]

        results: list[EvidenceChunk] = []

        for index in ranked_indices:
            chunk = self.chunks[int(index)]

            results.append(
                chunk.model_copy(
                    update={
                        "vector_score": float(scores[index]),
                    }
                )
            )

        return results

    @staticmethod
    def _normalise_embeddings(
        embeddings: Sequence[Sequence[float]],
        expected_rows: int,
    ) -> np.ndarray:
        """
        将向量转换成二维矩阵并执行 L2 归一化。
        """
        try:
            matrix = np.asarray(embeddings, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "Embedding response contains invalid values"
            ) from exc

        if matrix.ndim != 2:
            raise RuntimeError(
                "Embedding response must be a two-dimensional matrix"
            )

        if matrix.shape[0] != expected_rows:
            raise RuntimeError(
                "Embedding response row count is incorrect"
            )

        if matrix.shape[1] == 0:
            raise RuntimeError(
                "Embedding vectors must have at least one dimension"
            )

        if not np.isfinite(matrix).all():
            raise RuntimeError(
                "Embedding vectors contain non-finite values"
            )

        norms = np.linalg.norm(matrix, axis=1, keepdims=True)

        if np.any(norms == 0):
            raise RuntimeError(
                "Embedding vectors must not be zero vectors"
            )

        return matrix / norms