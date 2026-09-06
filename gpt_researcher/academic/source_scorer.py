from collections.abc import Sequence

from .schemas import PaperMetadata, SourceQualityScore


class SourceScorer:
    """
    使用可解释、确定性的规则评估学术来源质量。

    总分由以下部分组成：

    - 来源类型：30分
    - 可追溯性：25分
    - 证据可用性：20分
    - 元数据完整性：15分
    - 引用信号：10分

    已撤稿来源直接被拒绝。
    """

    SOURCE_TYPE_SCORES: dict[str, float] = {
        "journal": 30.0,
        "conference": 28.0,
        "preprint": 18.0,
        "report": 15.0,
        "web": 5.0,
        "unknown": 0.0,
    }

    SOURCE_TYPE_CAPS: dict[str, float] = {
        "journal": 100.0,
        "conference": 100.0,
        "preprint": 79.0,
        "report": 69.0,
        "web": 49.0,
        "unknown": 39.0,
    }

    def score(
        self,
        paper: PaperMetadata,
    ) -> SourceQualityScore:
        """
        对单个来源进行质量评分。
        """
        if paper.is_retracted is True:
            return SourceQualityScore(
                paper_id=paper.paper_id,
                total_score=0.0,
                grade="rejected",
                source_type_score=0.0,
                traceability_score=0.0,
                evidence_access_score=0.0,
                metadata_completeness_score=0.0,
                citation_score=0.0,
                score_cap=0.0,
                reasons=[
                    "Source is marked as retracted.",
                ],
            )

        reasons: list[str] = []

        source_type_score = self._score_source_type(
            paper,
            reasons,
        )
        traceability_score = self._score_traceability(
            paper,
            reasons,
        )
        evidence_access_score = self._score_evidence_access(
            paper,
            reasons,
        )
        metadata_score = self._score_metadata_completeness(
            paper,
            reasons,
        )
        citation_score = self._score_citations(
            paper,
            reasons,
        )

        raw_score = (
            source_type_score
            + traceability_score
            + evidence_access_score
            + metadata_score
            + citation_score
        )

        score_cap = self.SOURCE_TYPE_CAPS.get(
            paper.source_type,
            self.SOURCE_TYPE_CAPS["unknown"],
        )

        total_score = min(raw_score, score_cap)

        if total_score < raw_score:
            reasons.append(
                "Final score was capped because of the "
                f"{paper.source_type} source type."
            )

        if paper.is_retracted is False:
            reasons.append(
                "No retraction flag is present in the "
                "available metadata."
            )
        else:
            reasons.append(
                "Retraction status is unavailable."
            )

        return SourceQualityScore(
            paper_id=paper.paper_id,
            total_score=round(total_score, 2),
            grade=self._grade(total_score),
            source_type_score=source_type_score,
            traceability_score=traceability_score,
            evidence_access_score=evidence_access_score,
            metadata_completeness_score=metadata_score,
            citation_score=citation_score,
            score_cap=score_cap,
            reasons=reasons,
        )

    def score_many(
        self,
        papers: Sequence[PaperMetadata],
    ) -> list[SourceQualityScore]:
        """
        对多个来源评分，并按总分从高到低排列。
        """
        scores = [
            self.score(paper)
            for paper in papers
        ]

        return sorted(
            scores,
            key=lambda result: (
                -result.total_score,
                result.paper_id,
            ),
        )

    def _score_source_type(
        self,
        paper: PaperMetadata,
        reasons: list[str],
    ) -> float:
        score = self.SOURCE_TYPE_SCORES.get(
            paper.source_type,
            0.0,
        )

        descriptions = {
            "journal": "Journal article source type.",
            "conference": "Conference paper source type.",
            "preprint": (
                "Preprint source type; peer review is not confirmed."
            ),
            "report": "Report source type.",
            "web": (
                "Web source type receives limited academic credit."
            ),
            "unknown": "Source type is unknown.",
        }

        reasons.append(
            descriptions.get(
                paper.source_type,
                descriptions["unknown"],
            )
        )

        return score

    @staticmethod
    def _score_traceability(
        paper: PaperMetadata,
        reasons: list[str],
    ) -> float:
        score = 0.0

        if paper.doi:
            score += 10.0
            reasons.append("DOI is available.")

        if paper.url:
            score += 5.0
            reasons.append("Source URL is available.")

        if paper.authors:
            score += 4.0
            reasons.append("Author information is available.")

        if paper.year is not None:
            score += 3.0
            reasons.append("Publication year is available.")

        if paper.venue:
            score += 3.0
            reasons.append("Publication venue is available.")

        return score

    @staticmethod
    def _score_evidence_access(
        paper: PaperMetadata,
        reasons: list[str],
    ) -> float:
        score = 0.0

        if paper.abstract and paper.abstract.strip():
            score += 8.0
            reasons.append("Abstract evidence is available.")

        if paper.full_text_url:
            score += 8.0
            reasons.append("Full-text location is available.")

        if paper.is_open_access is True:
            score += 4.0
            reasons.append("Source is marked as open access.")

        return score

    @staticmethod
    def _score_metadata_completeness(
        paper: PaperMetadata,
        reasons: list[str],
    ) -> float:
        score = 0.0

        if paper.title.strip():
            score += 5.0

        if paper.paper_id.strip():
            score += 4.0

        if paper.venue:
            score += 3.0

        if paper.source_type != "unknown":
            score += 3.0

        if score == 15.0:
            reasons.append(
                "Core source metadata is complete."
            )

        return score

    @staticmethod
    def _score_citations(
        paper: PaperMetadata,
        reasons: list[str],
    ) -> float:
        citation_count = paper.citation_count

        if citation_count is None:
            reasons.append(
                "Citation count is unavailable."
            )
            return 0.0

        if citation_count >= 1000:
            score = 10.0
        elif citation_count >= 200:
            score = 8.0
        elif citation_count >= 50:
            score = 6.0
        elif citation_count >= 10:
            score = 4.0
        elif citation_count >= 1:
            score = 2.0
        else:
            score = 0.0

        reasons.append(
            f"Available citation count: {citation_count}."
        )

        return score

    @staticmethod
    def _grade(
        total_score: float,
    ) -> str:
        if total_score >= 80:
            return "high"

        if total_score >= 50:
            return "medium"

        return "low"