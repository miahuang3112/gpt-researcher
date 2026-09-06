#!/usr/bin/env python3
"""Apply evidence-bounded ChatGPT pre-labels to the RET-07 annotation sheet.

These labels are intentionally identified as AI pre-annotations.  They are useful
for pipeline dry-runs and comparison with an independent evaluator, but they must
not be represented as the independent human gold labels required by RET-07.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


EXPECTED_DIMENSIONS = {
    "research_question_clarity",
    "methodological_rigor",
    "experimental_design",
    "baseline_quality",
    "result_validity",
    "reproducibility",
    "novelty",
    "limitations_discussion",
    "citation_quality",
}

# Chunk numbers were selected after reading the complete 90-chunk paper corpus.
# Grade 2 = direct/substantive evidence; grade 1 = partial/supporting evidence.
LABELS: dict[str, dict[int, set[int]]] = {
    "research_question_clarity": {
        2: {2, 3, 4},
        1: {5, 16, 26},
    },
    "methodological_rigor": {
        2: {7, 8, 18, 86, 87, 89},
        1: {9, 10, 14, 17, 19, 22, 64, 66, 67, 69, 70, 71, 72, 74, 81, 85, 90},
    },
    "experimental_design": {
        2: {8, 9, 10, 18, 19, 22, 64, 66, 69, 84},
        1: {13, 14, 17, 55, 57, 59, 62, 63, 65, 67, 70, 71, 74, 76, 82, 90},
    },
    "baseline_quality": {
        2: {9, 11, 14, 15, 19, 57, 58, 73, 74},
        1: {10, 12, 16, 17, 20, 21, 22, 56, 59, 75, 78, 79, 80, 81},
    },
    "result_validity": {
        2: {10, 11, 12, 13, 15, 20, 21, 22, 65, 74, 84},
        1: {5, 16, 23, 56, 58, 59, 60, 61, 62, 63, 73, 75, 76, 77, 78, 79, 80, 81, 82, 83},
    },
    "reproducibility": {
        2: {8, 18, 22, 27, 64, 66, 67, 69, 71, 72, 84, 87, 89, 90},
        1: {9, 10, 13, 14, 17, 19, 57, 59, 62, 63, 70, 74, 81, 82, 85, 86},
    },
    "novelty": {
        2: {3, 4, 24, 25},
        1: {2, 7, 16, 26},
    },
    "limitations_discussion": {
        2: {8, 56, 57, 68, 74, 78, 81, 82, 88},
        1: {12, 13, 16, 17, 21, 23, 26, 60, 62, 65, 76, 79, 83, 84},
    },
    "citation_quality": {
        2: {24, 25, 74},
        1: {3, 7, 8, 9, 10, 16, 17, 18, 19, 56, 57, 59, 62, 63, 76, 87, 90},
    },
}

RATIONALES = {
    "research_question_clarity": {
        2: "片段直接陈述研究问题、动机、目标或工作范围，是定位研究问题的核心证据。",
        1: "片段补充研究目标、动机或范围，但不是最完整的问题陈述。",
        0: "片段没有实质说明研究问题、目标、范围或动机。",
    },
    "methodological_rigor": {
        2: "片段直接给出方法定义、算法步骤、关键假设或技术实现，是方法严谨性的核心证据。",
        1: "片段提供部分方法、评测器或实现细节，可辅助判断方法设计。",
        0: "片段没有提供可用于判断方法、假设、技术设计或实现的实质证据。",
    },
    "experimental_design": {
        2: "片段直接说明数据、评测协议、指标、样本或对照设计，是实验设计的核心证据。",
        1: "片段提供某项补充实验、消融、样本或评测设置，但覆盖不完整。",
        0: "片段没有提供数据集、协议、指标、消融、控制或样本量方面的实质信息。",
    },
    "baseline_quality": {
        2: "片段明确列出基线和比较结果，或解释了比较条件与公平性，是基线质量的核心证据。",
        1: "片段涉及基线、对照结果或比较条件，但对公平性和竞争性的支撑有限。",
        0: "片段没有提供可判断基线相关性、竞争性或比较公平性的实质证据。",
    },
    "result_validity": {
        2: "片段给出直接结果并包含验证、误差、人工对照或重要负面发现，是结论有效性的核心证据。",
        1: "片段报告辅助结果、趋势、负面结果或解释，但独立支撑结论的力度有限。",
        0: "片段没有提供支持或限制论文结论的结果、不确定性分析或负面发现。",
    },
    "reproducibility": {
        2: "片段给出复现实验所需的关键版本、流程、提示词、算法、超参数、硬件或样本设置。",
        1: "片段提供部分可复现细节，但不足以单独重建完整实验。",
        0: "片段没有提供数据、版本、随机性、超参数、硬件、代码或评测流程等复现信息。",
    },
    "novelty": {
        2: "片段明确将本文方法与既有工作区分，并说明新的技术贡献或未覆盖的应用范围。",
        1: "片段概括技术贡献或新用途，但与既有工作的区别论证较简略。",
        0: "片段没有说明本文相对既有工作的技术新意。",
    },
    "limitations_discussion": {
        2: "片段明确讨论假设、失败模式、偏差、泛化或可扩展性限制以及待解决问题。",
        1: "片段提到局部失败、代价、数据偏差、替代解释或未来工作，可作为辅助限制证据。",
        0: "片段没有承认与假设、失败案例、泛化、伦理或未解决问题相关的实质限制。",
    },
    "citation_quality": {
        2: "片段围绕关键主张综合并比较相关工作，清楚建立本文与文献的关系。",
        1: "片段将具体方法、数据集或判断与相关来源相连，但平衡性或综合程度有限。",
        0: "片段没有把重要主张与相关且平衡的既有工作实质连接起来。",
    },
}


def _chunk_number(chunk_id: str) -> int:
    match = re.search(r"-c(\d+)-", chunk_id)
    if not match:
        raise ValueError(f"Cannot parse chunk number: {chunk_id}")
    return int(match.group(1))


def _grade(dimension: str, chunk_number: int) -> int:
    by_grade = LABELS[dimension]
    if chunk_number in by_grade[2]:
        return 2
    if chunk_number in by_grade[1]:
        return 1
    return 0


def _note(row: dict[str, str], grade: int) -> str:
    reason = RATIONALES[row["dimension"]][grade]
    if row["dimension"] == "citation_quality" and row["section"] == "References":
        reason = "该片段仅为参考文献条目；书目存在本身不能证明重要主张获得了相关且平衡的文献支持。"
    return f"ChatGPT AI预标注（{grade}）：{reason}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def apply(input_path: Path) -> None:
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = list(reader)
        fieldnames = reader.fieldnames

    if not fieldnames or len(rows) != 810:
        raise ValueError(f"Expected 810 annotation rows, found {len(rows)}")
    if {row["dimension"] for row in rows} != EXPECTED_DIMENSIONS:
        raise ValueError("Annotation sheet dimensions do not match the RET-07 rubric")

    backup_path = input_path.with_name("annotation_tasks.human-unlabeled-backup.tsv")
    if not backup_path.exists():
        shutil.copy2(input_path, backup_path)

    counts: dict[str, Counter[int]] = defaultdict(Counter)
    chunks_per_dimension: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        chunk_number = _chunk_number(row["chunk_id"])
        grade = _grade(row["dimension"], chunk_number)
        row["relevance_grade"] = str(grade)
        row["annotator_notes"] = _note(row, grade)
        counts[row["dimension"]][grade] += 1
        chunks_per_dimension[row["dimension"]].add(chunk_number)

    for dimension in EXPECTED_DIMENSIONS:
        if len(chunks_per_dimension[dimension]) != 90:
            raise ValueError(f"{dimension} does not contain all 90 chunks")
        if counts[dimension][1] + counts[dimension][2] == 0:
            raise ValueError(f"{dimension} has no positive relevance labels")

    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=input_path.parent, delete=False
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
        temporary_path = Path(handle.name)
    temporary_path.replace(input_path)

    provenance = {
        "artifact": input_path.name,
        "paper_id": rows[0]["paper_id"],
        "annotation_type": "AI pre-annotation; not independent human gold labels",
        "annotator": "OpenAI ChatGPT/Codex session",
        "model_identifier": "not exposed by the session runtime",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rubric": {"0": "not relevant", "1": "partially relevant", "2": "directly relevant"},
        "method": "Full-paper chunk review followed by dimension-specific evidence grading",
        "row_count": len(rows),
        "sha256": _sha256(input_path),
        "backup": backup_path.name,
        "label_distribution": {
            dimension: {str(grade): counts[dimension][grade] for grade in (0, 1, 2)}
            for dimension in sorted(EXPECTED_DIMENSIONS)
        },
        "integrity_warning": (
            "Do not report these labels as RET-07 human annotation. Use independent human labels "
            "for unbiased gold evaluation and disclose any AI assistance."
        ),
    }
    provenance_path = input_path.with_name("annotation_tasks.chatgpt-provenance.json")
    provenance_path.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(json.dumps(provenance, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=Path(
            "evals/academic_retrieval/datasets/arxiv-2410.02902/annotation_tasks.tsv"
        ),
    )
    args = parser.parse_args()
    apply(args.path)


if __name__ == "__main__":
    main()
