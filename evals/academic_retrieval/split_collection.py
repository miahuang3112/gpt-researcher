from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .dataset import load_corpus_jsonl, load_queries_jsonl, validate_query_references
from .export_pdf_benchmark import render_jsonl


SPLIT_ROLES = ("development", "test")


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


def _validate_source_hashes(
    collection_dir: Path,
    collection_manifest: dict[str, Any],
) -> None:
    expected = {
        "corpus.jsonl": collection_manifest.get("corpus_sha256"),
        "queries.jsonl": collection_manifest.get("queries_sha256"),
    }
    for filename, expected_hash in expected.items():
        path = collection_dir / filename
        if not isinstance(expected_hash, str) or _sha256(path) != expected_hash:
            raise ValueError(f"Collection {filename} does not match its frozen manifest")


def _parse_assignments(
    spec: dict[str, Any],
    available_paper_ids: set[str],
) -> dict[str, list[str]]:
    assignments: dict[str, list[str]] = {}
    assigned: list[str] = []
    for role in SPLIT_ROLES:
        paper_ids = spec.get(role)
        if not isinstance(paper_ids, list) or not paper_ids:
            raise ValueError(f"split spec must contain a non-empty {role} list")
        if any(not isinstance(paper_id, str) or not paper_id.strip() for paper_id in paper_ids):
            raise ValueError(f"{role} paper IDs must be non-empty strings")
        if len(paper_ids) != len(set(paper_ids)):
            raise ValueError(f"{role} contains duplicate paper IDs")
        assignments[role] = paper_ids
        assigned.extend(paper_ids)

    overlap = set(assignments["development"]).intersection(assignments["test"])
    if overlap:
        raise ValueError(f"Development/test papers overlap: {sorted(overlap)}")
    if set(assigned) != available_paper_ids:
        missing = available_paper_ids.difference(assigned)
        unknown = set(assigned).difference(available_paper_ids)
        raise ValueError(
            "Split must assign every collection paper exactly once; "
            f"missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    return assignments


def split_collection(
    *,
    collection_dir: str | Path,
    spec_path: str | Path,
    output_dir: str | Path,
    overwrite: bool = False,
) -> dict[str, Path]:
    source_dir = Path(collection_dir).expanduser().resolve()
    resolved_spec = Path(spec_path).expanduser().resolve()
    resolved_output = Path(output_dir).expanduser().resolve()
    collection_manifest_path = source_dir / "collection_manifest.json"
    collection_manifest = _load_object(collection_manifest_path, "collection manifest")
    spec = _load_object(resolved_spec, "split spec")
    _validate_source_hashes(source_dir, collection_manifest)

    chunks = load_corpus_jsonl(source_dir / "corpus.jsonl")
    queries = load_queries_jsonl(source_dir / "queries.jsonl")
    validate_query_references(chunks, queries)
    available_paper_ids = {chunk.paper_id for chunk in chunks}
    assignments = _parse_assignments(spec, available_paper_ids)

    output_paths: dict[str, Path] = {
        role: resolved_output / role / "manifest.json" for role in SPLIT_ROLES
    }
    output_paths["split_manifest"] = resolved_output / "split_manifest.json"
    existing = [path for path in output_paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Split build would overwrite existing files: "
            + ", ".join(str(path) for path in existing)
        )

    member_by_paper_id = {
        member["paper_id"]: member
        for member in collection_manifest.get("members", [])
        if isinstance(member, dict) and isinstance(member.get("paper_id"), str)
    }
    split_records: dict[str, dict[str, Any]] = {}
    for role in SPLIT_ROLES:
        paper_ids = assignments[role]
        split_chunks = [chunk for chunk in chunks if chunk.paper_id in paper_ids]
        split_queries = [query for query in queries if query.paper_id in paper_ids]
        validate_query_references(split_chunks, split_queries)
        role_dir = resolved_output / role
        role_dir.mkdir(parents=True, exist_ok=True)
        corpus_path = role_dir / "corpus.jsonl"
        queries_path = role_dir / "queries.jsonl"
        _atomic_write(
            corpus_path,
            render_jsonl([
                chunk.model_dump(
                    mode="json",
                    exclude_none=True,
                    exclude={"bm25_score", "vector_score", "fused_score"},
                )
                for chunk in split_chunks
            ]),
        )
        _atomic_write(
            queries_path,
            render_jsonl([
                query.model_dump(mode="json", exclude_none=True)
                for query in split_queries
            ]),
        )
        manifest = {
            "format_version": 1,
            "split_name": spec.get("split_name", resolved_output.name),
            "split_role": role,
            "tuning_permitted": role == "development",
            "paper_ids": paper_ids,
            "paper_count": len(paper_ids),
            "chunk_count": len(split_chunks),
            "query_count": len(split_queries),
            "positive_judgement_count": sum(
                grade > 0
                for query in split_queries
                for grade in query.relevance_by_chunk_id.values()
            ),
            "corpus_sha256": _sha256(corpus_path),
            "queries_sha256": _sha256(queries_path),
            "parent_collection_manifest_sha256": _sha256(collection_manifest_path),
            "members": [member_by_paper_id[paper_id] for paper_id in paper_ids],
            "usage_policy": (
                "Parameter selection is allowed on this development split."
                if role == "development"
                else "Held-out test split: parameter selection is forbidden."
            ),
        }
        manifest_path = role_dir / "manifest.json"
        _atomic_write(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        split_records[role] = {
            "path": role,
            "manifest_sha256": _sha256(manifest_path),
            "paper_ids": paper_ids,
            "paper_count": manifest["paper_count"],
            "chunk_count": manifest["chunk_count"],
            "query_count": manifest["query_count"],
            "positive_judgement_count": manifest["positive_judgement_count"],
            "corpus_sha256": manifest["corpus_sha256"],
            "queries_sha256": manifest["queries_sha256"],
        }

    split_manifest = {
        "format_version": 1,
        "split_name": spec.get("split_name", resolved_output.name),
        "strategy": "fixed-paper-disjoint",
        "selection_note": spec.get("selection_note"),
        "spec": str(resolved_spec),
        "spec_sha256": _sha256(resolved_spec),
        "parent_collection": str(source_dir),
        "parent_collection_manifest_sha256": _sha256(collection_manifest_path),
        "splits": split_records,
        "integrity": {
            "paper_disjoint": True,
            "all_parent_papers_assigned_once": True,
            "test_tuning_permitted": False,
        },
    }
    resolved_output.mkdir(parents=True, exist_ok=True)
    _atomic_write(
        output_paths["split_manifest"],
        json.dumps(split_manifest, ensure_ascii=False, indent=2) + "\n",
    )
    return output_paths


def assert_tuning_manifest(manifest_path: str | Path) -> dict[str, Any]:
    resolved = Path(manifest_path).expanduser().resolve()
    manifest = _load_object(resolved, "split manifest")
    if manifest.get("split_role") != "development" or manifest.get("tuning_permitted") is not True:
        raise ValueError("Parameter tuning requires a development split manifest")
    for filename, key in (("corpus.jsonl", "corpus_sha256"), ("queries.jsonl", "queries_sha256")):
        if _sha256(resolved.parent / filename) != manifest.get(key):
            raise ValueError(f"Development {filename} does not match its frozen manifest")
    return manifest


def assert_test_manifest(manifest_path: str | Path) -> dict[str, Any]:
    resolved = Path(manifest_path).expanduser().resolve()
    manifest = _load_object(resolved, "split manifest")
    if manifest.get("split_role") != "test" or manifest.get("tuning_permitted") is not False:
        raise ValueError("Held-out evaluation requires a test split manifest")
    for filename, key in (("corpus.jsonl", "corpus_sha256"), ("queries.jsonl", "queries_sha256")):
        if _sha256(resolved.parent / filename) != manifest.get(key):
            raise ValueError(f"Test {filename} does not match its frozen manifest")
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create fixed paper-disjoint retrieval splits.")
    parser.add_argument("--collection-dir", required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    paths = split_collection(
        collection_dir=args.collection_dir,
        spec_path=args.spec,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )
    manifest = _load_object(paths["split_manifest"], "split manifest")
    development = manifest["splits"]["development"]
    test = manifest["splits"]["test"]
    print(
        "Built paper-disjoint splits: "
        f"development={development['paper_count']} papers/{development['query_count']} queries, "
        f"test={test['paper_count']} papers/{test['query_count']} queries."
    )
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
