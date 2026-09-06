import re
import unicodedata
from collections.abc import Sequence

from rank_bm25 import BM25Okapi

from gpt_researcher.academic.schemas import (
    EvidenceChunk,
)


class BM25Retriever:
    """
    使用Okapi BM25对EvidenceChunk进行关键词排序。

    该检索器不会联网搜索论文，只会对已经获得并切分好的
    证据片段进行排序。
    """

    TOKEN_PATTERN = re.compile(
        r"10\.\d{4,9}/[-._;()/:a-z0-9]+"
        r"|[a-z0-9]+(?:[._/-][a-z0-9]+)*",
        re.IGNORECASE,
    )

    def __init__(
        self,
        chunks: Sequence[EvidenceChunk],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if not chunks:
            raise ValueError(
                "BM25 requires at least one evidence chunk."
            )

        if k1 <= 0:
            raise ValueError(
                "BM25 k1 must be greater than 0."
            )

        if not 0 <= b <= 1:
            raise ValueError(
                "BM25 b must be between 0 and 1."
            )

        self.chunks = list(chunks)
        self.k1 = k1
        self.b = b

        self.tokenized_corpus = [
            self.tokenize(chunk.text)
            for chunk in self.chunks
        ]

        if not any(self.tokenized_corpus):
            raise ValueError(
                "BM25 corpus contains no searchable tokens."
            )

        self.vocabulary = {
            token
            for document in self.tokenized_corpus
            for token in document
        }

        self.index = BM25Okapi(
            self.tokenized_corpus,
            k1=self.k1,
            b=self.b,
        )

    @classmethod
    def tokenize(
        cls,
        text: str,
    ) -> list[str]:
        """
        统一大小写并保留DOI、模型名和数据集名等技术标识。
        """

        normalized_text = unicodedata.normalize(
            "NFKC",
            text,
        ).lower()

        tokens: list[str] = []

        for match in cls.TOKEN_PATTERN.finditer(
            normalized_text
        ):
            token = match.group(0).rstrip(
                ".,;:"
            )

            if not token:
                continue

            # 保留完整技术标识，例如：
            # qwen3-embedding-0.6b
            tokens.append(token)

            # 同时加入模型名的组成部分，使不同写法也能匹配。
            # DOI保持为完整token，避免破坏精确匹配。
            if not token.startswith("10."):
                components = re.split(
                    r"[._/-]+",
                    token,
                )

                tokens.extend(
                    component
                    for component in components
                    if component
                    and component != token
                )

        return tokens

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[EvidenceChunk]:
        """
        返回BM25得分最高的证据片段。
        """

        query = query.strip()

        if not query:
            raise ValueError(
                "BM25 query cannot be empty."
            )

        if top_k < 1:
            raise ValueError(
                "BM25 top_k must be at least 1."
            )

        query_tokens = self.tokenize(query)

        if not query_tokens:
            return []

        # 查询词与语料库完全没有交集时，不返回无关片段。
        if not (
            set(query_tokens)
            & self.vocabulary
        ):
            return []

        scores = self.index.get_scores(
            query_tokens
        )

        ranked_indices = sorted(
            range(len(self.chunks)),
            key=lambda index: (
                -float(scores[index]),
                index,
            ),
        )

        selected_indices = ranked_indices[
            :min(top_k, len(ranked_indices))
        ]

        return [
            self.chunks[index].model_copy(
                update={
                    "bm25_score": float(
                        scores[index]
                    )
                }
            )
            for index in selected_indices
        ]