#!/usr/bin/env python3
"""Apply ChatGPT-reviewed relevance labels to the RET-07B expansion papers.

The mappings in this file were produced by reading every exported chunk. They
are AI annotations requested by the project owner, not independent human gold
labels. The generated provenance files preserve that distinction.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from evals.academic_retrieval.apply_chatgpt_annotations import RATIONALES


DATASET_ROOT = Path("evals/academic_retrieval/datasets")

# grade 2 = directly/substantively relevant; grade 1 = partial/supporting.
LABELS: dict[str, dict[str, dict[int, set[int]]]] = {
    "arxiv-1706.03762": {
        "research_question_clarity": {2: {2, 3}, 1: {4, 16}},
        "methodological_rigor": {2: {5, 6, 7, 8}, 1: {9, 10, 13, 14}},
        "experimental_design": {2: {11, 12, 13, 14, 15}, 1: {2, 8, 10}},
        "baseline_quality": {2: {11, 15}, 1: {12, 13, 14}},
        "result_validity": {2: {11, 12, 13, 15}, 1: {2, 14, 22, 23, 24}},
        "reproducibility": {2: {5, 6, 7, 8, 11, 12, 13, 14}, 1: {10}},
        "novelty": {2: {3, 4, 16}, 1: {2, 8, 9}},
        "limitations_discussion": {2: {10, 16}, 1: {4, 8, 9, 14}},
        "citation_quality": {2: {3, 4}, 1: {5, 8, 9, 11, 14, 15, 16}},
    },
    "arxiv-1810.04805": {
        "research_question_clarity": {2: {1, 2, 3}, 1: {22}},
        "methodological_rigor": {
            2: {5, 6, 7, 8, 9, 10, 12, 15, 16, 18, 35, 36},
            1: {3, 11, 13, 17, 21, 34, 37, 43, 44},
        },
        "experimental_design": {
            2: {10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 35, 36, 42, 43, 44},
            1: {9, 31, 34, 38, 41},
        },
        "baseline_quality": {
            2: {11, 13, 14, 15, 16, 17, 20, 21, 34, 37},
            1: {10, 12, 18, 19, 36},
        },
        "result_validity": {
            2: {11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 43, 44},
            1: {1, 12, 22},
        },
        "reproducibility": {
            2: {7, 8, 9, 10, 12, 15, 16, 35, 36, 38, 41, 43},
            1: {4, 5, 6, 11, 13, 14, 17, 18, 20, 21, 32, 42, 44},
        },
        "novelty": {2: {1, 2, 3, 4, 34, 37}, 1: {5, 6, 7, 8, 22}},
        "limitations_discussion": {
            2: {12, 13, 17, 19, 43},
            1: {3, 16, 18, 20, 22, 36, 42, 44},
        },
        "citation_quality": {2: {2, 3, 4, 6, 34, 37}, 1: {10, 13, 20, 21, 32, 38}},
    },
    "arxiv-2005.11401": {
        "research_question_clarity": {2: {2, 3, 5}, 1: {21}},
        "methodological_rigor": {2: {6, 7, 8, 9}, 1: {4, 10, 37, 39}},
        "experimental_design": {
            2: {10, 11, 13, 14, 16, 17, 18, 37, 38, 40},
            1: {12, 15, 39},
        },
        "baseline_quality": {2: {12, 13, 14, 17}, 1: {10, 11, 16, 18, 20, 37, 40}},
        "result_validity": {2: {12, 13, 14, 16, 17, 18}, 1: {15, 21, 39, 40}},
        "reproducibility": {2: {6, 7, 8, 9, 10, 11, 37, 38, 40}, 1: {13, 14, 16, 17, 18, 39}},
        "novelty": {2: {2, 3, 5, 19, 20}, 1: {4, 6, 21}},
        "limitations_discussion": {2: {11, 18, 22, 39}, 1: {3, 9, 14, 16, 20, 21, 38, 40}},
        "citation_quality": {2: {3, 5, 19, 20}, 1: {6, 8, 9, 10, 11, 12, 22}},
    },
    "arxiv-2106.09685": {
        "research_question_clarity": {2: {2, 3, 4, 5, 21}, 1: {26}},
        "methodological_rigor": {
            2: {5, 6, 9, 10, 21, 22, 23, 24, 25, 55, 65},
            1: {7, 8, 11, 38, 48, 63},
        },
        "experimental_design": {
            2: {11, 12, 13, 14, 15, 16, 17, 18, 22, 23, 24, 25, 36, 37, 39, 40, 41, 42, 43, 44, 45, 46, 47, 49, 50, 52, 54, 56, 57, 59, 62, 66, 67},
            1: {8, 21, 48, 51, 53, 58, 60, 61, 63, 64},
        },
        "baseline_quality": {
            2: {12, 13, 14, 15, 16, 17, 18, 36, 37, 38, 41, 42, 50, 52, 56, 57},
            1: {7, 8, 19, 20, 43, 44, 45, 47, 48, 49, 54, 59, 62, 67},
        },
        "result_validity": {
            2: {12, 13, 15, 16, 17, 18, 22, 23, 24, 25, 37, 41, 42, 50, 52, 54, 56, 57, 62, 66, 67},
            1: {2, 8, 14, 21, 26, 27, 36, 38, 48, 53, 58, 60, 61, 63, 64, 65},
        },
        "reproducibility": {
            2: {8, 10, 11, 14, 16, 17, 21, 22, 23, 24, 25, 39, 40, 41, 42, 43, 44, 45, 46, 47, 49, 55, 59, 62, 65},
            1: {5, 6, 9, 12, 13, 15, 18, 36, 37, 38, 48, 50, 54, 56, 57, 63, 66, 67},
        },
        "novelty": {2: {2, 3, 4, 5, 9, 10, 18, 19, 20, 21, 26}, 1: {6, 7, 8, 11, 23, 24, 25, 38}},
        "limitations_discussion": {
            2: {4, 7, 11, 21, 22, 24, 26, 27, 54, 62, 67},
            1: {8, 18, 19, 20, 23, 25, 36, 38, 48, 53, 58, 60, 61, 63, 65},
        },
        "citation_quality": {2: {4, 7, 18, 19, 20}, 1: {3, 5, 8, 14, 16, 30, 31, 32, 33, 34, 38, 39, 40, 41, 42, 45}},
    },
}


def _chunk_number(chunk_id: str) -> int:
    match = re.search(r"-c(\d+)-", chunk_id)
    if not match:
        raise ValueError(f"Cannot parse chunk number: {chunk_id}")
    return int(match.group(1))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _grade(labels: dict[int, set[int]], chunk_number: int) -> int:
    if chunk_number in labels[2]:
        return 2
    if chunk_number in labels[1]:
        return 1
    return 0


def apply_dataset(dataset_name: str, labels: dict[str, dict[int, set[int]]]) -> None:
    dataset_dir = DATASET_ROOT / dataset_name
    annotation_path = dataset_dir / "annotation_tasks.tsv"
    with annotation_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = reader.fieldnames
        rows = list(reader)

    if not fieldnames or not rows:
        raise ValueError(f"Empty annotation sheet: {annotation_path}")
    dimensions = {row["dimension"] for row in rows}
    if dimensions != set(labels):
        raise ValueError(f"Dimension mismatch for {dataset_name}")

    chunk_ids = {row["chunk_id"] for row in rows}
    expected_rows = len(chunk_ids) * len(dimensions)
    if len(rows) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows for {dataset_name}, found {len(rows)}")

    backup = dataset_dir / "annotation_tasks.unlabeled-backup.tsv"
    if not backup.exists():
        shutil.copy2(annotation_path, backup)

    counts: dict[str, Counter[int]] = defaultdict(Counter)
    for row in rows:
        chunk_number = _chunk_number(row["chunk_id"])
        grade = _grade(labels[row["dimension"]], chunk_number)
        reason = RATIONALES[row["dimension"]][grade]
        if row["dimension"] == "citation_quality" and row["section"] == "References":
            reason = "该片段仅为参考文献条目；书目存在本身不能证明正文的重要主张获得相关且平衡的支持。"
        row["relevance_grade"] = str(grade)
        row["annotator_notes"] = f"ChatGPT AI标注与复核（{grade}）：{reason}"
        counts[row["dimension"]][grade] += 1

    if any(counts[dimension][1] + counts[dimension][2] == 0 for dimension in dimensions):
        raise ValueError(f"A query has no positive evidence in {dataset_name}")

    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=dataset_dir, delete=False
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
        temporary_path = Path(handle.name)
    temporary_path.replace(annotation_path)

    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    provenance = {
        "artifact": annotation_path.name,
        "paper_id": manifest["paper_id"],
        "annotation_type": "ChatGPT AI annotation and ChatGPT review",
        "annotator": "OpenAI ChatGPT/Codex session",
        "model_identifier": "not exposed by the session runtime",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "review_status": "AI_review_completed",
        "independent_human_annotation": False,
        "row_count": len(rows),
        "chunk_count": len(chunk_ids),
        "query_count": len(dimensions),
        "annotation_sha256": _sha256(annotation_path),
        "unlabeled_backup": backup.name,
        "rubric": {"0": "irrelevant", "1": "partially relevant", "2": "directly/highly relevant"},
        "label_distribution": {
            dimension: {str(grade): counts[dimension][grade] for grade in (0, 1, 2)}
            for dimension in sorted(dimensions)
        },
        "integrity_warning": "These labels are AI-generated and AI-reviewed, not independent human gold labels.",
    }
    provenance_path = dataset_dir / "annotation_tasks.chatgpt-review-provenance.json"
    provenance_path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(provenance, ensure_ascii=False))


def main() -> None:
    for dataset_name, labels in LABELS.items():
        apply_dataset(dataset_name, labels)


if __name__ == "__main__":
    main()
