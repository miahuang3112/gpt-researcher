from __future__ import annotations

import json

import pytest

from evals.academic_retrieval.build_collection import build_collection


def _write_dataset(root, paper_id: str, chunk_id: str) -> None:
    root.mkdir(parents=True)
    (root / "corpus.jsonl").write_text(
        json.dumps({"chunk_id": chunk_id, "paper_id": paper_id, "text": "evidence"}) + "\n",
        encoding="utf-8",
    )
    (root / "queries.jsonl").write_text(
        json.dumps(
            {
                "query_id": f"{paper_id}-q",
                "paper_id": paper_id,
                "query": "Where is the evidence?",
                "dimension": "reproducibility",
                "relevance_by_chunk_id": {chunk_id: 2},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "paper_id": paper_id,
                "source_document": f"{paper_id}.pdf",
                "source_sha256": f"hash-{paper_id}",
            }
        ),
        encoding="utf-8",
    )
    (root / "provenance.json").write_text(
        json.dumps(
            {
                "annotation_type": "AI annotation",
                "independent_human_annotation": False,
            }
        ),
        encoding="utf-8",
    )


def test_build_collection_combines_and_hashes_members(tmp_path) -> None:
    _write_dataset(tmp_path / "p1", "paper-1", "c1")
    _write_dataset(tmp_path / "p2", "paper-2", "c2")
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(
        json.dumps(
            {
                "collection_name": "test-collection",
                "members": [
                    {"dataset_dir": "p1", "annotation_provenance": "provenance.json"},
                    {"dataset_dir": "p2", "annotation_provenance": "provenance.json"},
                ],
            }
        ),
        encoding="utf-8",
    )

    paths = build_collection(spec_path=spec_path, output_dir=tmp_path / "output")
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))

    assert manifest["paper_count"] == 2
    assert manifest["chunk_count"] == 2
    assert manifest["query_count"] == 2
    assert manifest["positive_judgement_count"] == 2
    assert len(manifest["corpus_sha256"]) == 64
    assert len(manifest["queries_sha256"]) == 64
    assert [member["paper_id"] for member in manifest["members"]] == ["paper-1", "paper-2"]


def test_build_collection_rejects_duplicate_paper_ids(tmp_path) -> None:
    _write_dataset(tmp_path / "p1", "paper-1", "c1")
    _write_dataset(tmp_path / "p2", "paper-1", "c2")
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(
        json.dumps(
            {
                "members": [
                    {"dataset_dir": "p1", "annotation_provenance": "provenance.json"},
                    {"dataset_dir": "p2", "annotation_provenance": "provenance.json"},
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate paper_id"):
        build_collection(spec_path=spec_path, output_dir=tmp_path / "output")
