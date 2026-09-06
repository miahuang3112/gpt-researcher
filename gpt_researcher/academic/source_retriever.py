import html
import os
import re
from typing import Any

import httpx
from dotenv import load_dotenv

from gpt_researcher.academic.schemas import PaperMetadata
from gpt_researcher.academic.paper_identity import (
    canonical_paper_id,
    deduplicate_papers,
    normalize_doi,
)


load_dotenv()

def _non_empty_string(
    value: Any,
) -> str | None:
    if isinstance(value, str):
        cleaned = value.strip()

        if cleaned:
            return cleaned

    return None


def _optional_bool(
    value: Any,
) -> bool | None:
    if isinstance(value, bool):
        return value

    return None


def _non_negative_int(
    value: Any,
) -> int | None:
    # bool是int的子类，因此必须单独排除。
    if isinstance(value, bool):
        return None

    if isinstance(value, int) and value >= 0:
        return value

    return None


def _extract_openalex_quality_metadata(
    work: dict[str, Any],
) -> dict[str, Any]:
    """
    从OpenAlex Work对象中提取来源质量评分需要的字段。
    """
    open_access = work.get("open_access")

    if not isinstance(open_access, dict):
        open_access = {}

    best_oa_location = work.get("best_oa_location")

    if not isinstance(best_oa_location, dict):
        best_oa_location = {}

    primary_location = work.get("primary_location")

    if not isinstance(primary_location, dict):
        primary_location = {}

    full_text_url = (
        _non_empty_string(
            best_oa_location.get("pdf_url")
        )
        or _non_empty_string(
            primary_location.get("pdf_url")
        )
        or _non_empty_string(
            work.get("content_url")
        )
    )

    # oa_url可能是PDF，也可能是开放访问落地页。
    # 只有OpenAlex明确标记为OA时才把它作为全文位置使用。
    if (
        full_text_url is None
        and open_access.get("is_oa") is True
    ):
        full_text_url = _non_empty_string(
            open_access.get("oa_url")
        )

    return {
        "citation_count": _non_negative_int(
            work.get("cited_by_count")
        ),
        "is_retracted": _optional_bool(
            work.get("is_retracted")
        ),
        "is_open_access": _optional_bool(
            open_access.get("is_oa")
        ),
        "full_text_url": full_text_url,
    }


def _extract_crossref_quality_metadata(
    work: dict[str, Any],
) -> dict[str, Any]:
    """
    从Crossref Work对象中提取来源质量评分需要的字段。

    Crossref搜索结果没有适合直接转换成is_retracted的
    统一布尔字段，因此这里保守地返回None。
    """
    full_text_url: str | None = None

    links = work.get("link")

    if isinstance(links, list):
        for link in links:
            if not isinstance(link, dict):
                continue

            content_type = str(
                link.get("content-type", "")
            ).lower()

            url = _non_empty_string(
                link.get("URL")
            )

            if url and "pdf" in content_type:
                full_text_url = url
                break

    is_open_access: bool | None = None
    licenses = work.get("license")

    open_license_markers = (
        "creativecommons.org/licenses/",
        "creativecommons.org/publicdomain/",
    )

    if isinstance(licenses, list):
        for license_record in licenses:
            if not isinstance(license_record, dict):
                continue

            license_url = _non_empty_string(
                license_record.get("URL")
            )

            if license_url is None:
                continue

            normalised_url = license_url.lower()

            if any(
                marker in normalised_url
                for marker in open_license_markers
            ):
                is_open_access = True
                break

    return {
        "citation_count": _non_negative_int(
            work.get("is-referenced-by-count")
        ),
        "is_retracted": None,
        "is_open_access": is_open_access,
        "full_text_url": full_text_url,
    }

class OpenAlexRetriever:
    """
    使用 OpenAlex API 搜索论文，并把返回结果转换成 PaperMetadata。
    """

    BASE_URL = "https://api.openalex.org/works"

    def __init__(
        self,
        api_key: str | None = None,
        timeout: float = 20.0,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENALEX_API_KEY")
        self.timeout = timeout

        if not self.api_key:
            raise ValueError(
                "OPENALEX_API_KEY is missing. "
                "Add it to the project .env file."
            )

    async def search(
        self,
        query: str,
        limit: int = 10,
    ) -> list[PaperMetadata]:
        """
        根据关键词搜索论文。
        """

        query = query.strip()

        if not query:
            raise ValueError("Search query cannot be empty.")

        if limit < 1:
            raise ValueError("Search limit must be at least 1.")

        params = {
            "search": query,
            "per-page": min(limit, 100),
            "api_key": self.api_key,
        }

        try:
            async with httpx.AsyncClient(
                timeout=self.timeout,
                follow_redirects=True,
            ) as client:
                response = await client.get(
                    self.BASE_URL,
                    params=params,
                )

        except httpx.RequestError as exc:
            raise RuntimeError(
                "Could not connect to OpenAlex."
            ) from exc

        if response.is_error:
            raise RuntimeError(
                f"OpenAlex returned HTTP {response.status_code}. "
                "Check whether OPENALEX_API_KEY is valid."
            )

        data = response.json()
        works = data.get("results", [])

        return deduplicate_papers(
            [self._to_metadata(work) for work in works]
        )[:limit]

    @classmethod
    def _to_metadata(
        cls,
        work: dict[str, Any],
    ) -> PaperMetadata:
        """
        把一条 OpenAlex Work 转换成统一的数据结构。
        """

        primary_location = work.get("primary_location") or {}
        source = primary_location.get("source") or {}
        open_access = work.get("open_access") or {}

        doi = cls._normalise_doi(work.get("doi"))
        openalex_id = (work.get("id") or "").rsplit("/", 1)[-1]
        title = work.get("title") or "Untitled"

        authors = [
            authorship["author"]["display_name"]
            for authorship in work.get("authorships", [])
            if authorship.get("author", {}).get("display_name")
        ]

        return PaperMetadata(
            paper_id=canonical_paper_id(
                doi=doi,
                title=title,
                fallback_id=openalex_id,
            ),
            source_ids=[openalex_id] if openalex_id else [],
            title=title,
            authors=authors,
            year=work.get("publication_year"),
            doi=doi,
            venue=source.get("display_name"),
            url=(
                primary_location.get("landing_page_url")
                or work.get("id")
            ),
            abstract=cls._reconstruct_abstract(
                work.get("abstract_inverted_index")
            ),
            source_type=cls._map_source_type(
                work_type=work.get("type"),
                source_type=source.get("type"),
            ),
            retrieved_from="openalex",
            **_extract_openalex_quality_metadata(work),
        )

    @staticmethod
    def _normalise_doi(doi: str | None) -> str | None:
        """
        把 DOI URL 转换为标准 DOI 字符串。
        """

        return normalize_doi(doi)

    @staticmethod
    def _reconstruct_abstract(
        inverted_index: dict[str, list[int]] | None,
    ) -> str | None:
        """
        把 OpenAlex 的倒排索引摘要恢复成普通文本。
        """

        if not inverted_index:
            return None

        positioned_words = [
            (position, word)
            for word, positions in inverted_index.items()
            for position in positions
        ]

        positioned_words.sort(key=lambda item: item[0])

        return " ".join(
            word for _, word in positioned_words
        )

    @staticmethod
    def _map_source_type(
        work_type: str | None,
        source_type: str | None,
    ) -> str:
        """
        把 OpenAlex 类型映射成项目内部类型。
        """

        work_type = (work_type or "").lower()
        source_type = (source_type or "").lower()

        if work_type == "preprint":
            return "preprint"

        if source_type == "conference" or work_type == "proceedings-article":
            return "conference"

        if source_type == "journal":
            return "journal"

        if work_type in {"book", "book-chapter", "reference-entry"}:
            return "book"

        if work_type == "report":
            return "report"

        return "unknown"

class CrossrefRetriever:
    """
    使用Crossref API搜索论文，并转换成PaperMetadata。
    """

    BASE_URL = "https://api.crossref.org/works"

    def __init__(
        self,
        mailto: str | None = None,
        timeout: float = 20.0,
    ) -> None:
        self.mailto = mailto or os.getenv("CROSSREF_MAILTO")
        self.timeout = timeout

        if not self.mailto:
            raise ValueError(
                "CROSSREF_MAILTO is missing. "
                "Add it to the project .env file."
            )

    async def search(
        self,
        query: str,
        limit: int = 10,
    ) -> list[PaperMetadata]:
        """
        根据论文题目、作者或其他文献信息搜索Crossref。
        """

        query = query.strip()

        if not query:
            raise ValueError("Search query cannot be empty.")

        if limit < 1:
            raise ValueError("Search limit must be at least 1.")

        params = {
            "query.bibliographic": query,
            "rows": min(limit, 1000),
            "mailto": self.mailto,
        }

        headers = {
            "User-Agent": (
                "academic-paper-review/0.1 "
                f"(mailto:{self.mailto})"
            )
        }

        try:
            async with httpx.AsyncClient(
                timeout=self.timeout,
                follow_redirects=True,
                headers=headers,
            ) as client:
                response = await client.get(
                    self.BASE_URL,
                    params=params,
                )

        except httpx.RequestError as exc:
            raise RuntimeError(
                "Could not connect to Crossref."
            ) from exc

        if response.is_error:
            raise RuntimeError(
                f"Crossref returned HTTP "
                f"{response.status_code}."
            )

        try:
            data = response.json()
            items = data["message"]["items"]

        except (ValueError, KeyError, TypeError) as exc:
            raise RuntimeError(
                "Crossref returned an invalid response."
            ) from exc

        return deduplicate_papers(
            [self._to_metadata(item) for item in items]
        )[:limit]

    @classmethod
    def _to_metadata(
        cls,
        item: dict[str, Any],
    ) -> PaperMetadata:
        """
        把一条Crossref记录转换成统一数据结构。
        """

        doi = cls._normalise_doi(item.get("DOI"))
        title = cls._first_value(
            item.get("title")
        ) or "Untitled"

        venue = cls._first_value(
            item.get("container-title")
        )

        authors = cls._extract_authors(
            item.get("author", [])
        )

        url = item.get("URL")

        source_id = doi or url or title

        return PaperMetadata(
            paper_id=canonical_paper_id(
                doi=doi,
                title=title,
                fallback_id=source_id,
            ),
            source_ids=[source_id] if source_id else [],
            title=title,
            authors=authors,
            year=cls._extract_year(item),
            doi=doi,
            venue=venue,
            url=url,
            abstract=cls._clean_abstract(
                item.get("abstract")
            ),
            source_type=cls._map_source_type(
                item.get("type")
            ),
            retrieved_from="crossref",
            **_extract_crossref_quality_metadata(item),
        )

    @staticmethod
    def _first_value(
        value: list[str] | None,
    ) -> str | None:
        if not value:
            return None

        first = value[0]

        if not isinstance(first, str):
            return None

        first = first.strip()

        return first or None

    @staticmethod
    def _extract_authors(
        authors: list[dict[str, Any]],
    ) -> list[str]:
        names: list[str] = []

        for author in authors:
            parts = [
                author.get("given"),
                author.get("family"),
            ]

            name = " ".join(
                part.strip()
                for part in parts
                if isinstance(part, str)
                and part.strip()
            )

            if name:
                names.append(name)

        return names

    @staticmethod
    def _extract_year(
        item: dict[str, Any],
    ) -> int | None:
        date_fields = [
            "published-print",
            "published-online",
            "published",
            "issued",
        ]

        for field in date_fields:
            date_data = item.get(field) or {}
            date_parts = date_data.get("date-parts") or []

            if date_parts and date_parts[0]:
                year = date_parts[0][0]

                if isinstance(year, int):
                    return year

        return None

    @staticmethod
    def _clean_abstract(
        abstract: str | None,
    ) -> str | None:
        """
        删除Crossref摘要中的HTML或JATS标签。
        """

        if not abstract:
            return None

        text = html.unescape(abstract)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        return text or None

    @staticmethod
    def _normalise_doi(
        doi: str | None,
    ) -> str | None:
        return normalize_doi(doi)

    @staticmethod
    def _map_source_type(
        crossref_type: str | None,
    ) -> str:
        crossref_type = (
            crossref_type or ""
        ).lower()

        if crossref_type == "journal-article":
            return "journal"

        if crossref_type in {
            "proceedings",
            "proceedings-article",
        }:
            return "conference"

        if crossref_type == "posted-content":
            return "preprint"

        if crossref_type in {
            "book",
            "book-chapter",
            "reference-book",
            "reference-entry",
        }:
            return "book"

        if crossref_type == "report":
            return "report"

        return "unknown"
