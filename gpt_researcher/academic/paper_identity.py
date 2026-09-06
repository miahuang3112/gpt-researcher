from __future__ import annotations

import hashlib
import html
import re
import unicodedata
from collections.abc import Sequence
from urllib.parse import unquote

from gpt_researcher.academic.schemas import PaperMetadata


DOI_PATTERN = re.compile(r"^10\.\d{4,9}/\S+$")
DOI_PREFIX_PATTERN = re.compile(
    r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)",
    flags=re.IGNORECASE,
)
TITLE_TOKEN_PATTERN = re.compile(r"[^\w]+", flags=re.UNICODE)
PLACEHOLDER_TITLES = {"", "untitled"}


def normalize_doi(value: str | None) -> str | None:
    """Return a comparable DOI string or ``None`` for invalid input."""
    if not isinstance(value, str):
        return None

    normalized = html.unescape(unquote(value)).strip()
    normalized = DOI_PREFIX_PATTERN.sub("", normalized).strip()
    normalized = normalized.split("?", 1)[0].split("#", 1)[0]
    normalized = normalized.rstrip(".,;").casefold()

    if not DOI_PATTERN.fullmatch(normalized):
        return None

    return normalized


def normalize_title(value: str | None) -> str:
    """Normalize Unicode, punctuation and whitespace for title matching."""
    if not isinstance(value, str):
        return ""

    normalized = unicodedata.normalize("NFKC", html.unescape(value))
    normalized = TITLE_TOKEN_PATTERN.sub(" ", normalized.casefold())
    return " ".join(normalized.split())


def canonical_paper_id(
    *,
    doi: str | None,
    title: str | None,
    fallback_id: str | None = None,
) -> str:
    """Build a deterministic identity, preferring DOI over normalized title."""
    normalized_doi = normalize_doi(doi)

    if normalized_doi:
        return normalized_doi

    normalized_title = normalize_title(title)

    if normalized_title not in PLACEHOLDER_TITLES:
        digest = hashlib.sha256(
            normalized_title.encode("utf-8")
        ).hexdigest()[:24]
        return f"title:{digest}"

    normalized_fallback = " ".join((fallback_id or "").split())

    if not normalized_fallback:
        raise ValueError(
            "A canonical paper identity requires a DOI, title, or fallback ID."
        )

    digest = hashlib.sha256(
        normalized_fallback.casefold().encode("utf-8")
    ).hexdigest()[:24]
    return f"source:{digest}"


def canonicalize_paper(paper: PaperMetadata) -> PaperMetadata:
    """Normalize identity fields while retaining source record identifiers."""
    normalized_doi = normalize_doi(paper.doi)
    canonical_id = canonical_paper_id(
        doi=normalized_doi,
        title=paper.title,
        fallback_id=paper.paper_id,
    )
    source_ids = _deduplicate_strings(
        [*paper.source_ids, paper.paper_id]
    )

    return paper.model_copy(
        update={
            "paper_id": canonical_id,
            "doi": normalized_doi,
            "title": " ".join(paper.title.split()) or "Untitled",
            "source_ids": source_ids,
        }
    )


def deduplicate_papers(
    papers: Sequence[PaperMetadata],
) -> list[PaperMetadata]:
    """
    Deduplicate DOI records first, then conservatively attach DOI-less title
    matches only when that title maps to one unambiguous DOI group.
    """
    canonical = [canonicalize_paper(paper) for paper in papers]
    doi_groups: dict[str, list[tuple[int, PaperMetadata]]] = {}
    title_only_groups: dict[str, list[tuple[int, PaperMetadata]]] = {}
    fallback_groups: dict[str, list[tuple[int, PaperMetadata]]] = {}

    for index, paper in enumerate(canonical):
        title_key = normalize_title(paper.title)

        if paper.doi:
            doi_groups.setdefault(paper.doi, []).append((index, paper))
        elif title_key not in PLACEHOLDER_TITLES:
            title_only_groups.setdefault(title_key, []).append((index, paper))
        else:
            fallback_groups.setdefault(paper.paper_id, []).append((index, paper))

    title_to_dois: dict[str, set[str]] = {}

    for doi, records in doi_groups.items():
        for _, paper in records:
            title_key = normalize_title(paper.title)
            if title_key not in PLACEHOLDER_TITLES:
                title_to_dois.setdefault(title_key, set()).add(doi)

    for title_key, records in list(title_only_groups.items()):
        matching_dois = title_to_dois.get(title_key, set())
        if len(matching_dois) == 1:
            doi = next(iter(matching_dois))
            doi_groups[doi].extend(records)
            del title_only_groups[title_key]

    groups = [
        records
        for mapping in (doi_groups, title_only_groups, fallback_groups)
        for records in mapping.values()
    ]
    groups.sort(key=lambda records: min(index for index, _ in records))

    return [
        _merge_group([paper for _, paper in records])
        for records in groups
    ]


def _merge_group(papers: Sequence[PaperMetadata]) -> PaperMetadata:
    primary = max(papers, key=_metadata_quality_key)
    doi = next((paper.doi for paper in papers if paper.doi), None)
    best_title = max(
        (paper.title for paper in papers),
        key=lambda value: (
            normalize_title(value) not in PLACEHOLDER_TITLES,
            len(value),
        ),
    )
    source_ids = _deduplicate_strings(
        source_id
        for paper in papers
        for source_id in paper.source_ids
    )

    values = {
        "paper_id": canonical_paper_id(
            doi=doi,
            title=best_title,
            fallback_id=primary.paper_id,
        ),
        "title": best_title,
        "doi": doi,
        "authors": _deduplicate_strings(
            author for paper in papers for author in paper.authors
        ),
        "source_ids": source_ids,
        "year": _first_present(papers, "year"),
        "venue": _first_present(papers, "venue"),
        "url": _first_present(papers, "url"),
        "abstract": _longest_present(papers, "abstract"),
        "citation_count": max(
            (
                paper.citation_count
                for paper in papers
                if paper.citation_count is not None
            ),
            default=None,
        ),
        "is_retracted": _merge_bool(papers, "is_retracted"),
        "is_open_access": _merge_bool(papers, "is_open_access"),
        "full_text_url": _first_present(papers, "full_text_url"),
    }
    return primary.model_copy(update=values)


def _metadata_quality_key(paper: PaperMetadata) -> tuple[int, int, int]:
    populated = sum(
        value not in (None, "", [], "unknown")
        for value in (
            paper.doi,
            paper.authors,
            paper.year,
            paper.venue,
            paper.url,
            paper.abstract,
            paper.citation_count,
            paper.full_text_url,
            paper.source_type,
        )
    )
    return (populated, len(paper.abstract or ""), len(paper.title))


def _first_present(
    papers: Sequence[PaperMetadata],
    field: str,
) -> object | None:
    for paper in sorted(papers, key=_metadata_quality_key, reverse=True):
        value = getattr(paper, field)
        if value not in (None, ""):
            return value
    return None


def _longest_present(
    papers: Sequence[PaperMetadata],
    field: str,
) -> str | None:
    values = [
        value
        for paper in papers
        if isinstance((value := getattr(paper, field)), str) and value.strip()
    ]
    return max(values, key=len, default=None)


def _merge_bool(
    papers: Sequence[PaperMetadata],
    field: str,
) -> bool | None:
    values = [getattr(paper, field) for paper in papers]
    if True in values:
        return True
    if False in values:
        return False
    return None


def _deduplicate_strings(values) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()

    for value in values:
        if not isinstance(value, str):
            continue
        cleaned = " ".join(value.split())
        key = cleaned.casefold()
        if cleaned and key not in seen:
            seen.add(key)
            result.append(cleaned)

    return result
