from __future__ import annotations

import json

import pytest

from evals.academic_retrieval.split_collection import (
    assert_test_manifest,
    assert_tuning_manifest,
    split_collection,
)


def _write_collection(root) -> None:
    root.mkdir()
    corpus_records = [
        {"chunk_id": "c1", "paper_id": "paper-1", "text": "first"},
        {"chunk_id": "c2", "paper_id": "paper-2", "text": "second"},
    ]
    query_records = [
        {
            "query_id": "q1",
            "paper_id": "paper-1",
            "query": "first?",
            "relevance_by_chunk_id": {"c1": 2},
        },
        {
            "query_id": "q2",
            "paper_id": "paper-2",
            "query": "second?",
            "relevance_by_chunk_id": {"c2": 2},
        },
    ]
    corpus_path = root / "corpus.jsonl"
    queries_path = root / "queries.jsonl"
    corpus_path.write_text("".join(json.dumps(row) + "\n" for row in corpus_records), encoding="utf-8")
    queries_path.write_text("".join(json.dumps(row) + "\n" for row in query_records), encoding="utf-8")
    import hashlib

    (root / "collection_manifest.json").write_text(
        json.dumps(
            {
                "corpus_sha256": hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
                "queries_sha256": hashlib.sha256(queries_path.read_bytes()).hexdigest(),
                "members": [
                    {"paper_id": "paper-1", "annotation_type": "AI"},
                    {"paper_id": "paper-2", "annotation_type": "AI"},
                ],
            }
        ),
        encoding="utf-8",
    )


def _write_spec(path, development, test) -> None:
    path.write_text(
        json.dumps({"split_name": "test-split", "development": development, "test": test}),
        encoding="utf-8",
    )


def test_split_collection_is_paper_disjoint_and_marks_usage(tmp_path) -> None:
    collection = tmp_path / "collection"
    _write_collection(collection)
    spec = tmp_path / "spec.json"
    _write_spec(spec, ["paper-1"], ["paper-2"])

    paths = split_collection(
        collection_dir=collection,
        spec_path=spec,
        output_dir=tmp_path / "split",
    )

    development = json.loads(paths["development"].read_text(encoding="utf-8"))
    test = json.loads(paths["test"].read_text(encoding="utf-8"))
    overall = json.loads(paths["split_manifest"].read_text(encoding="utf-8"))
    assert development["paper_ids"] == ["paper-1"]
    assert development["tuning_permitted"] is True
    assert test["paper_ids"] == ["paper-2"]
    assert test["tuning_permitted"] is False
    assert overall["integrity"]["paper_disjoint"] is True
    assert_tuning_manifest(paths["development"])
    assert_test_manifest(paths["test"])
    with pytest.raises(ValueError, match="development split"):
        assert_tuning_manifest(paths["test"])
    with pytest.raises(ValueError, match="test split"):
        assert_test_manifest(paths["development"])


def test_split_collection_rejects_overlap(tmp_path) -> None:
    collection = tmp_path / "collection"
    _write_collection(collection)
    spec = tmp_path / "spec.json"
    _write_spec(spec, ["paper-1"], ["paper-1", "paper-2"])

    with pytest.raises(ValueError, match="overlap"):
        split_collection(
            collection_dir=collection,
            spec_path=spec,
            output_dir=tmp_path / "split",
        )


def test_split_collection_rejects_unassigned_paper(tmp_path) -> None:
    collection = tmp_path / "collection"
    _write_collection(collection)
    spec = tmp_path / "spec.json"
    _write_spec(spec, ["paper-1"], ["unknown-paper"])

    with pytest.raises(ValueError, match="assign every collection paper"):
        split_collection(
            collection_dir=collection,
            spec_path=spec,
            output_dir=tmp_path / "split",
        )


def test_tuning_manifest_rejects_mutated_development_data(tmp_path) -> None:
    collection = tmp_path / "collection"
    _write_collection(collection)
    spec = tmp_path / "spec.json"
    _write_spec(spec, ["paper-1"], ["paper-2"])
    paths = split_collection(
        collection_dir=collection,
        spec_path=spec,
        output_dir=tmp_path / "split",
    )
    development_corpus = paths["development"].parent / "corpus.jsonl"
    development_corpus.write_text(development_corpus.read_text() + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="frozen manifest"):
        assert_tuning_manifest(paths["development"])
