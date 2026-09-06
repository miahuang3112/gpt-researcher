from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .dataset import (
    load_corpus_jsonl,
    load_queries_jsonl,
    validate_query_references,
)
from .export_pdf_benchmark import render_jsonl


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_object(path: Path, description: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not load {description} from {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{description} must be a JSON object: {path}")
    return payload


def _atomic_write(path: Path, content: str) -> None:
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(content, encoding="utf-8")
    temporary_path.replace(path)


def build_collection(
    *,
    spec_path: str | Path,
    output_dir: str | Path,
    overwrite: bool = False,
) -> dict[str, Path]:
    resolved_spec = Path(spec_path).expanduser().resolve()
    resolved_output = Path(output_dir).expanduser().resolve()
    spec = _load_object(resolved_spec, "collection spec")
    members = spec.get("members")
    if not isinstance(members, list) or len(members) < 2:
        raise ValueError("collection spec must contain at least two members")

    paths = {
        "corpus": resolved_output / "corpus.jsonl",
        "queries": resolved_output / "queries.jsonl",
        "manifest": resolved_output / "collection_manifest.json",
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Collection build would overwrite existing files: "
            + ", ".join(str(path) for path in existing)
        )

    all_chunks = []
    all_queries = []
    member_records: list[dict[str, Any]] = []
    seen_paper_ids: set[str] = set()
    seen_chunk_ids: set[str] = set()
    seen_query_ids: set[str] = set()

    for index, raw_member in enumerate(members, start=1):
        if not isinstance(raw_member, dict):
            raise ValueError(f"collection member {index} must be an object")
        dataset_value = raw_member.get("dataset_dir")
        provenance_value = raw_member.get("annotation_provenance")
        if not isinstance(dataset_value, str) or not dataset_value.strip():
            raise ValueError(f"collection member {index} has no dataset_dir")
        if not isinstance(provenance_value, str) or not provenance_value.strip():
            raise ValueError(f"collection member {index} has no annotation_provenance")

        dataset_dir = (resolved_spec.parent / dataset_value).resolve()
        corpus_path = dataset_dir / "corpus.jsonl"
        queries_path = dataset_dir / "queries.jsonl"
        manifest_path = dataset_dir / "manifest.json"
        provenance_path = dataset_dir / provenance_value
        manifest = _load_object(manifest_path, "dataset manifest")
        provenance = _load_object(provenance_path, "annotation provenance")
        chunks = load_corpus_jsonl(corpus_path)
        queries = load_queries_jsonl(queries_path)
        validate_query_references(chunks, queries)

        paper_ids = {chunk.paper_id for chunk in chunks}
        if len(paper_ids) != 1:
            raise ValueError(f"Dataset must contain exactly one paper: {dataset_dir}")
        paper_id = next(iter(paper_ids))
        if manifest.get("paper_id") != paper_id:
            raise ValueError(f"Manifest paper_id mismatch for {dataset_dir}")
        if paper_id in seen_paper_ids:
            raise ValueError(f"Duplicate paper_id in collection: {paper_id}")

        chunk_ids = {chunk.chunk_id for chunk in chunks}
        query_ids = {query.query_id for query in queries}
        duplicate_chunks = seen_chunk_ids.intersection(chunk_ids)
        duplicate_queries = seen_query_ids.intersection(query_ids)
        if duplicate_chunks:
            raise ValueError(f"Duplicate chunk IDs: {sorted(duplicate_chunks)}")
        if duplicate_queries:
            raise ValueError(f"Duplicate query IDs: {sorted(duplicate_queries)}")

        seen_paper_ids.add(paper_id)
        seen_chunk_ids.update(chunk_ids)
        seen_query_ids.update(query_ids)
        all_chunks.extend(chunks)
        all_queries.extend(queries)
        member_records.append(
            {
                "paper_id": paper_id,
                "dataset_dir": dataset_value,
                "chunk_count": len(chunks),
                "query_count": len(queries),
                "positive_judgement_count": sum(
                    grade > 0
                    for query in queries
                    for grade in query.relevance_by_chunk_id.values()
                ),
                "source_document": manifest.get("source_document"),
                "source_sha256": manifest.get("source_sha256"),
                "corpus_sha256": _sha256(corpus_path),
                "queries_sha256": _sha256(queries_path),
                "annotation_provenance": provenance_value,
                "annotation_provenance_sha256": _sha256(provenance_path),
                "annotation_type": provenance.get("annotation_type"),
                "independent_human_annotation": provenance.get(
                    "independent_human_annotation",
                    provenance.get("annotation_lineage", {}).get(
                        "independent_blind_annotation"
                    ),
                ),
            }
        )

    corpus_records = [
        chunk.model_dump(
            mode="json",
            exclude_none=True,
            exclude={"bm25_score", "vector_score", "fused_score"},
        )
        for chunk in all_chunks
    ]
    query_records = [
        query.model_dump(mode="json", exclude_none=True)
        for query in all_queries
    ]
    rendered_corpus = render_jsonl(corpus_records)
    rendered_queries = render_jsonl(query_records)
    resolved_output.mkdir(parents=True, exist_ok=True)
    _atomic_write(paths["corpus"], rendered_corpus)
    _atomic_write(paths["queries"], rendered_queries)

    manifest = {
        "format_version": 1,
        "collection_name": spec.get("collection_name", resolved_output.name),
        "spec": str(resolved_spec),
        "spec_sha256": _sha256(resolved_spec),
        "paper_count": len(seen_paper_ids),
        "chunk_count": len(all_chunks),
        "query_count": len(all_queries),
        "positive_judgement_count": sum(
            grade > 0
            for query in all_queries
            for grade in query.relevance_by_chunk_id.values()
        ),
        "corpus_sha256": _sha256(paths["corpus"]),
        "queries_sha256": _sha256(paths["queries"]),
        "members": member_records,
        "integrity_warning": (
            "Annotation provenance differs by member. Inspect member metadata "
            "before making human-gold or inter-annotator claims."
        ),
    }
    _atomic_write(
        paths["manifest"],
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    )
    return paths


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a validated multi-paper retrieval collection.")
    parser.add_argument("--spec", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    paths = build_collection(
        spec_path=args.spec,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )
    manifest = _load_object(paths["manifest"], "collection manifest")
    print(
        f"Built {manifest['paper_count']} papers, {manifest['chunk_count']} chunks, "
        f"{manifest['query_count']} queries and "
        f"{manifest['positive_judgement_count']} positive judgements."
    )
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
