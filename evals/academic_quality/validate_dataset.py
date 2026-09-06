from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from gpt_researcher.academic.paper_evaluator import REQUIRED_DIMENSIONS


QUALITY_DIR = Path(__file__).resolve().parent
RETRIEVAL_DATASETS_DIR = QUALITY_DIR.parent / "academic_retrieval" / "datasets"


class QualityDatasetError(ValueError):
    """The versioned paper-quality reference set is invalid."""


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise QualityDatasetError(
                    f"Invalid JSON in {path} line {line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise QualityDatasetError(
                    f"Expected an object in {path} line {line_number}."
                )
            rows.append(row)
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_pdf_path(source_document: str) -> Path:
    standard = QUALITY_DIR.parent / "academic_retrieval" / "sources" / source_document
    if standard.exists():
        return standard
    desktop_copy = QUALITY_DIR.parents[2] / source_document
    if desktop_copy.exists():
        return desktop_copy
    raise QualityDatasetError(f"Source PDF not found: {source_document}")


def validate_dataset() -> dict[str, Any]:
    rubric = _load_json(QUALITY_DIR / "rubric.v1.json")
    paper_document = _load_json(QUALITY_DIR / "papers.v1.json")
    provenance = _load_json(QUALITY_DIR / "provenance.chatgpt.v1.json")
    labels = _load_jsonl(QUALITY_DIR / "labels.chatgpt.v1.jsonl")

    expected_dimensions = list(REQUIRED_DIMENSIONS)
    if list(rubric.get("dimensions", {})) != expected_dimensions:
        raise QualityDatasetError(
            "Rubric dimensions or ordering do not match production dimensions."
        )

    if provenance.get("annotator_type") != "ai":
        raise QualityDatasetError("ChatGPT provenance must declare annotator_type=ai.")
    if provenance.get("independent_human_review") is not False:
        raise QualityDatasetError(
            "The dataset must not claim independent human review."
        )

    papers = paper_document.get("papers")
    if not isinstance(papers, list) or len(papers) != 5:
        raise QualityDatasetError("papers.v1.json must contain exactly five papers.")
    papers_by_id = {paper["paper_id"]: paper for paper in papers}
    if len(papers_by_id) != len(papers):
        raise QualityDatasetError("Paper IDs must be unique.")
    if {paper["split_role"] for paper in papers} != {"development", "test"}:
        raise QualityDatasetError("Both development and test papers are required.")

    corpus_chunk_ids: dict[str, set[str]] = {}
    for paper_id, paper in papers_by_id.items():
        dataset_name = paper_id.replace(":", "-")
        manifest = _load_json(RETRIEVAL_DATASETS_DIR / dataset_name / "manifest.json")
        if manifest["source_sha256"] != paper["source_sha256"]:
            raise QualityDatasetError(f"Manifest hash mismatch for {paper_id}.")
        pdf_path = _source_pdf_path(paper["source_document"])
        if _sha256(pdf_path) != paper["source_sha256"]:
            raise QualityDatasetError(f"Source PDF hash mismatch for {paper_id}.")
        corpus = _load_jsonl(RETRIEVAL_DATASETS_DIR / dataset_name / "corpus.jsonl")
        corpus_chunk_ids[paper_id] = {row["chunk_id"] for row in corpus}

    if len(labels) != len(papers):
        raise QualityDatasetError("There must be one label record per paper.")
    seen_papers: set[str] = set()
    label_count = 0
    for record in labels:
        paper_id = record.get("paper_id")
        if paper_id not in papers_by_id or paper_id in seen_papers:
            raise QualityDatasetError(f"Unknown or duplicate label paper: {paper_id}")
        seen_papers.add(paper_id)
        if record.get("annotator_type") != "ai":
            raise QualityDatasetError(f"{paper_id} must declare annotator_type=ai.")
        dimensions = record.get("dimensions")
        if not isinstance(dimensions, list):
            raise QualityDatasetError(f"Missing dimensions for {paper_id}.")
        names = [item.get("dimension") for item in dimensions]
        if names != expected_dimensions:
            raise QualityDatasetError(
                f"Dimension coverage or ordering is invalid for {paper_id}."
            )
        for item in dimensions:
            label_count += 1
            score = item.get("score")
            if isinstance(score, bool) or not isinstance(score, int) or not 1 <= score <= 5:
                raise QualityDatasetError(
                    f"Invalid score for {paper_id}/{item.get('dimension')}."
                )
            confidence = item.get("confidence")
            if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
                raise QualityDatasetError(
                    f"Invalid confidence for {paper_id}/{item.get('dimension')}."
                )
            evidence = item.get("evidence_chunk_ids")
            if not isinstance(evidence, list) or not evidence:
                raise QualityDatasetError(
                    f"Missing evidence for {paper_id}/{item.get('dimension')}."
                )
            invalid_evidence = set(evidence) - corpus_chunk_ids[paper_id]
            if invalid_evidence:
                raise QualityDatasetError(
                    f"Unknown evidence for {paper_id}: {sorted(invalid_evidence)}"
                )
            for field in ("rationale", "strengths", "concerns"):
                if not item.get(field):
                    raise QualityDatasetError(
                        f"Missing {field} for {paper_id}/{item.get('dimension')}."
                    )

    if seen_papers != set(papers_by_id):
        raise QualityDatasetError("Not every paper has a label record.")
    if label_count != 45:
        raise QualityDatasetError(f"Expected 45 dimension labels, found {label_count}.")

    return {
        "paper_count": len(papers),
        "dimension_count": len(expected_dimensions),
        "label_count": label_count,
        "development_papers": sum(p["split_role"] == "development" for p in papers),
        "test_papers": sum(p["split_role"] == "test" for p in papers),
        "annotator_type": "ai",
        "independent_human_review": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate the versioned ChatGPT paper-quality reference set."
    )
    parser.parse_args()
    print(json.dumps(validate_dataset(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
