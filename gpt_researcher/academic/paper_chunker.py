from __future__ import annotations

import hashlib
import re
from collections import Counter
from collections.abc import Sequence
from math import ceil
from dataclasses import dataclass
from pathlib import Path

import fitz
import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from gpt_researcher.academic.schemas import EvidenceChunk
from gpt_researcher.academic.content_security import (
    detect_untrusted_content_warnings,
)


@dataclass(frozen=True)
class PaperPage:
    """一页论文文本。页码从 1 开始。"""

    page_number: int
    text: str


class AcademicPaperChunker:
    """
    把学术论文 PDF 切成可追踪的 EvidenceChunk。

    切块原则：
    1. 不跨页，保留准确页码；
    2. 优先按章节、段落和句子边界切分；
    3. 使用 token 数，而不是字符数；
    4. 长文本约 800 tokens，每块重叠约 120 tokens；
    5. 短章节不会被直接丢弃。
    """

    SECTION_ALIASES = {
        "abstract": "Abstract",
        "摘要": "Abstract",
        "introduction": "Introduction",
        "引言": "Introduction",
        "background": "Background",
        "related work": "Related Work",
        "literature review": "Related Work",
        "相关工作": "Related Work",
        "method": "Methods",
        "methods": "Methods",
        "methodology": "Methods",
        "materials and methods": "Methods",
        "方法": "Methods",
        "experimental setup": "Experiments",
        "experiment": "Experiments",
        "experiments": "Experiments",
        "实验": "Experiments",
        "result": "Results",
        "results": "Results",
        "结果": "Results",
        "discussion": "Discussion",
        "讨论": "Discussion",
        "results and discussion": "Results and Discussion",
        "limitations": "Limitations",
        "局限性": "Limitations",
        "conclusion": "Conclusion",
        "conclusions": "Conclusion",
        "结论": "Conclusion",
        "references": "References",
        "bibliography": "References",
        "参考文献": "References",
        "appendix": "Appendix",
        "附录": "Appendix",
        "appendices": "Appendix",
    "supplementary material": "Appendix",
    "supplemental material": "Appendix",
    "补充材料": "Appendix",
        "acknowledgements": "Acknowledgements",
        "acknowledgments": "Acknowledgements",
    }

    HEADING_PREFIX_PATTERN = re.compile(
        r"^(?:\d+(?:\.\d+)*|[ivxlcdm]+)[.)]?\s+",
        flags=re.IGNORECASE,
    )

    APPENDIX_PREFIX_PATTERN = re.compile(
        r"^(?:APPENDIX(?:\s+[A-Z])?|[A-Z](?:\.\d+)*)"
        r"(?:[.)])?\s+",
        flags=re.IGNORECASE,
    )

    MARGIN_LINE_COUNT = 3
    MIN_REPEATED_MARGIN_PAGES = 3
    REPEATED_MARGIN_RATIO = 0.30

    def __init__(
        self,
        target_tokens: int = 800,
        overlap_tokens: int = 120,
        min_chunk_tokens: int = 100,
        encoding_name: str = "cl100k_base",
    ) -> None:
        if target_tokens <= 0:
            raise ValueError("target_tokens must be greater than 0")

        if overlap_tokens < 0 or overlap_tokens >= target_tokens:
            raise ValueError(
                "overlap_tokens must be at least 0 and smaller than target_tokens"
            )

        if min_chunk_tokens <= 0 or min_chunk_tokens > target_tokens:
            raise ValueError(
                "min_chunk_tokens must be greater than 0 "
                "and no larger than target_tokens"
            )

        self.target_tokens = target_tokens
        self.overlap_tokens = overlap_tokens
        self.min_chunk_tokens = min_chunk_tokens
        self.encoding = tiktoken.get_encoding(encoding_name)

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=target_tokens,
            chunk_overlap=overlap_tokens,
            length_function=self.count_tokens,
            keep_separator=True,
            separators=[
                "\n\n",
                "\n",
                "。", "！", "？", "；",
                ". ", "! ", "? ", "; ",
                "，", ", ",
                " ",
                "",
            ],
        )

    def count_tokens(self, text: str) -> int:
        """计算文本的近似 token 数。"""

        return len(
            self.encoding.encode(
                text,
                disallowed_special=(),
            )
        )

    def chunk_pdf(
        self,
        file_path: str | Path,
        paper_id: str,
        source_url: str | None = None,
    ) -> list[EvidenceChunk]:
        """读取本地 PDF，并转换成 EvidenceChunk。"""

        pdf_path = Path(file_path).expanduser().resolve()

        if not pdf_path.is_file():
            raise FileNotFoundError(f"PDF file does not exist: {pdf_path}")

        pages: list[PaperPage] = []

        with fitz.open(str(pdf_path)) as document:
            for page_index, page in enumerate(document):
                pages.append(
                    PaperPage(
                        page_number=page_index + 1,
                        text=page.get_text("text", sort=True) or "",
                    )
                )

        return self.chunk_pages(
            pages=pages,
            paper_id=paper_id,
            source_url=source_url or str(pdf_path),
        )

    def chunk_pages(
        self,
        pages: Sequence[PaperPage],
        paper_id: str,
        source_url: str | None = None,
    ) -> list[EvidenceChunk]:
        """
        把已经提取好的页面文本转换成 EvidenceChunk。

        这个方法可以脱离真实 PDF 做单元测试。
        """

        normalized_paper_id = paper_id.strip()

        if not normalized_paper_id:
            raise ValueError("paper_id cannot be empty")

        evidence_chunks: list[EvidenceChunk] = []
        current_section: str | None = None
        chunk_number = 0

        cleaned_pages = self._remove_repeated_marginal_lines(
            pages
        )

        for page in cleaned_pages:
            if page.page_number < 1:
                raise ValueError("page_number must start from 1")

            page_sections, current_section = self._extract_page_sections(
                text=page.text,
                current_section=current_section,
            )

            for section, section_text in page_sections:
                split_texts = self._split_section_text(section_text)

                for chunk_text in split_texts:
                    chunk_number += 1

                    chunk_id = self._create_chunk_id(
                        paper_id=normalized_paper_id,
                        page_number=page.page_number,
                        section=section,
                        chunk_number=chunk_number,
                        text=chunk_text,
                    )

                    evidence_chunks.append(
                        EvidenceChunk(
                            chunk_id=chunk_id,
                            paper_id=normalized_paper_id,
                            text=chunk_text,
                            section=section,
                            page=page.page_number,
                            source_url=source_url,
                            content_warnings=(
                                detect_untrusted_content_warnings(chunk_text)
                            ),
                        )
                    )

        if not evidence_chunks:
            raise ValueError(
                "The PDF contains no extractable text. "
                "It may be an image-only PDF and require OCR."
            )

        return evidence_chunks

    def _remove_repeated_marginal_lines(
        self,
        pages: Sequence[PaperPage],
    ) -> list[PaperPage]:
        """
        删除多页PDF中重复出现的页眉和页脚。

        只检查每页最前面和最后面的少量非空行，
        避免删除正文中正常重复的句子。
        """

        page_list = list(pages)

        if len(page_list) < self.MIN_REPEATED_MARGIN_PAGES:
            return page_list

        normalized_counts: Counter[str] = Counter()
        page_line_data: list[list[str]] = []

        for page in page_list:
            nonempty_lines = [
                line.strip()
                for line in page.text.splitlines()
                if line.strip()
            ]

            page_line_data.append(nonempty_lines)

            marginal_lines = (
                nonempty_lines[:self.MARGIN_LINE_COUNT]
                + nonempty_lines[-self.MARGIN_LINE_COUNT:]
            )

            # 同一行在同一页最多计数一次。
            normalized_candidates = {
                self._normalize_margin_line(line)
                for line in marginal_lines
                if 5 <= len(line) <= 200
            }

            normalized_counts.update(normalized_candidates)

        repetition_threshold = max(
            self.MIN_REPEATED_MARGIN_PAGES,
            ceil(
                len(page_list)
                * self.REPEATED_MARGIN_RATIO
            ),
        )

        repeated_marginal_lines = {
            line
            for line, count in normalized_counts.items()
            if count >= repetition_threshold
        }

        if not repeated_marginal_lines:
            return page_list

        cleaned_pages: list[PaperPage] = []

        for page in page_list:
            raw_lines = page.text.splitlines()

            nonempty_indexes = [
                index
                for index, line in enumerate(raw_lines)
                if line.strip()
            ]

            marginal_indexes = set(
                nonempty_indexes[:self.MARGIN_LINE_COUNT]
                + nonempty_indexes[-self.MARGIN_LINE_COUNT:]
            )

            kept_lines: list[str] = []

            for index, line in enumerate(raw_lines):
                normalized_line = self._normalize_margin_line(
                    line
                )

                is_repeated_margin = (
                    index in marginal_indexes
                    and normalized_line
                    in repeated_marginal_lines
                )

                if not is_repeated_margin:
                    kept_lines.append(line)

            cleaned_pages.append(
                PaperPage(
                    page_number=page.page_number,
                    text="\n".join(kept_lines).strip(),
                )
            )

        return cleaned_pages

    @staticmethod
    def _normalize_margin_line(line: str) -> str:
        """标准化页眉和页脚文本，供跨页比较。"""

        return " ".join(line.split()).casefold()
    
    def _extract_page_sections(
        self,
        text: str,
        current_section: str | None,
    ) -> tuple[list[tuple[str | None, str]], str | None]:
        """识别一页中的标准论文章节，并保留跨页章节状态。"""

        sections: list[tuple[str | None, str]] = []
        active_section = current_section
        buffer: list[str] = []

        for raw_line in text.splitlines():
            line = raw_line.strip()
            detected_section = self._detect_section_heading(
                line,
                current_section=active_section,
            )
            if detected_section is not None:
                buffered_text = "\n".join(buffer).strip()

                if buffered_text:
                    sections.append((active_section, buffered_text))

                buffer.clear()
                active_section = detected_section
                continue

            buffer.append(raw_line)

        remaining_text = "\n".join(buffer).strip()

        if remaining_text:
            sections.append((active_section, remaining_text))

        return sections, active_section

    def _detect_section_heading(self, line: str, current_section: str | None = None) -> str | None:
        """把“2 Methods”等标题转换为统一章节名称。"""

        normalized_line = " ".join(line.split())

        if not normalized_line:
            return None

        if len(normalized_line) > 80:
            return None

        if self._is_appendix_heading(normalized_line,current_section=current_section):
            return "Appendix"

        without_number = self.HEADING_PREFIX_PATTERN.sub(
            "",
            normalized_line,
        )

        candidate = without_number.casefold().strip(" .:：-—–")

        return self.SECTION_ALIASES.get(candidate)

    def _is_appendix_heading(self, line: str, current_section: str | None = None) -> bool:
        """
        识别以下附录标题：

        Appendix A
        A MBR DECODING
        A.2 COMPARISON WITH UNIVERSAL SELF-CONSISTENCY
        B.1 ADDITIONAL EXPERIMENTS
        """

        match = self.APPENDIX_PREFIX_PATTERN.match(line)

        if match is None:
            return False

        if line.casefold().startswith("appendix"):
            return True

        if current_section not in {"References", "Appendix"}:
            return False

        if len(line.split()) > 16:
            return False

        if line.endswith((".", "?", "!", "。", "？", "！")):
            return False

        prefix = match.group(0).strip()

        # A.2、B.1等编号通常明确表示附录子章节。
        if "." in prefix:
            return True

        alphabetic_characters = [
            character
            for character in line
            if character.isalpha()
        ]

        if not alphabetic_characters:
            return False

        uppercase_count = sum(
            character.isupper()
            for character in alphabetic_characters
        )

        uppercase_ratio = (
            uppercase_count
            / len(alphabetic_characters)
        )

        # 防止把“A model improves...”这种普通正文误认为附录。
        return uppercase_ratio >= 0.65

    def _split_section_text(self, text: str) -> list[str]:
        """在一个页面和章节内部进行 token-aware 切块。"""

        chunks = [
            chunk.strip()
            for chunk in self.text_splitter.split_text(text)
            if chunk.strip()
        ]

        if len(chunks) < 2:
            return chunks

        last_chunk = chunks[-1]

        if self.count_tokens(last_chunk) >= self.min_chunk_tokens:
            return chunks

        combined = self._join_without_duplicate(
            chunks[-2],
            last_chunk,
        )

        maximum_merged_size = (
            self.target_tokens + self.min_chunk_tokens
        )

        if self.count_tokens(combined) <= maximum_merged_size:
            chunks[-2:] = [combined]

        return chunks

    @staticmethod
    def _join_without_duplicate(left: str, right: str) -> str:
        """合并短尾块，同时去掉 overlap 产生的重复内容。"""

        maximum_overlap = min(len(left), len(right))

        for overlap_size in range(maximum_overlap, 0, -1):
            if left.endswith(right[:overlap_size]):
                return left + right[overlap_size:]

        return f"{left}\n{right}"

    @staticmethod
    def _create_chunk_id(
        paper_id: str,
        page_number: int,
        section: str | None,
        chunk_number: int,
        text: str,
    ) -> str:
        """根据来源和正文生成稳定、可复现的 chunk ID。"""

        identity = (
            f"{paper_id}\0{page_number}\0{section or ''}"
            f"\0{chunk_number}\0{text}"
        )

        digest = hashlib.sha256(
            identity.encode("utf-8")
        ).hexdigest()[:16]

        return f"{paper_id}-p{page_number}-c{chunk_number}-{digest}"
