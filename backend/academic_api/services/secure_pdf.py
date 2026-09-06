from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO
from uuid import uuid4

import fitz


PDF_SIGNATURE = b"%PDF-"
COPY_CHUNK_SIZE = 1024 * 1024


class PdfSecurityError(ValueError):
    """A submitted file failed a bounded, public-safe PDF check."""


@dataclass(frozen=True)
class StoredPdf:
    path: Path
    size_bytes: int
    page_count: int
    sha256: str


def _store_pdf_stream(
    source_file: BinaryIO,
    *,
    storage_dir: str | Path,
    max_bytes: int,
    max_pages: int,
    max_extracted_characters: int,
) -> StoredPdf:
    if max_bytes <= 0 or max_pages <= 0 or max_extracted_characters <= 0:
        raise ValueError("PDF limits must be greater than zero.")

    destination_dir = Path(storage_dir).expanduser().resolve()
    destination_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(destination_dir, 0o700)
    temporary_path = destination_dir / f".{uuid4().hex}.uploading"
    final_path = destination_dir / f"{uuid4().hex}.pdf"
    digest = hashlib.sha256()
    size_bytes = 0

    try:
        signature = source_file.read(len(PDF_SIGNATURE))
        if signature != PDF_SIGNATURE:
            raise PdfSecurityError("File signature is not PDF.")

        with temporary_path.open("xb") as destination_file:
            os.chmod(temporary_path, 0o600)
            digest.update(signature)
            destination_file.write(signature)
            size_bytes = len(signature)
            if size_bytes > max_bytes:
                raise PdfSecurityError("PDF exceeds the size limit.")

            while chunk := source_file.read(COPY_CHUNK_SIZE):
                size_bytes += len(chunk)
                if size_bytes > max_bytes:
                    raise PdfSecurityError("PDF exceeds the size limit.")
                digest.update(chunk)
                destination_file.write(chunk)

        try:
            with fitz.open(str(temporary_path)) as document:
                if not document.is_pdf:
                    raise PdfSecurityError("File is not a valid PDF.")
                if document.needs_pass:
                    raise PdfSecurityError("Encrypted PDFs are not supported.")
                page_count = document.page_count
                if document.embfile_names():
                    raise PdfSecurityError(
                        "PDF attachments or active content are not supported."
                    )

                extracted_characters = 0
                for page in document:
                    extracted_characters += len(page.get_text("text") or "")
                    if extracted_characters > max_extracted_characters:
                        raise PdfSecurityError(
                            "PDF extracted text exceeds the safety limit."
                        )

                for xref in range(1, document.xref_length()):
                    obj = document.xref_object(xref, compressed=False)
                    if any(
                        marker in obj
                        for marker in (
                            "/S/JavaScript",
                            "/S /JavaScript",
                            "/S/Launch",
                            "/S /Launch",
                        )
                    ):
                        raise PdfSecurityError(
                            "PDF attachments or active content are not supported."
                        )
        except PdfSecurityError:
            raise
        except Exception as exc:
            raise PdfSecurityError("PDF is malformed or unreadable.") from exc

        if page_count < 1:
            raise PdfSecurityError("PDF has no pages.")
        if page_count > max_pages:
            raise PdfSecurityError("PDF exceeds the page limit.")

        temporary_path.replace(final_path)
        return StoredPdf(
            path=final_path,
            size_bytes=size_bytes,
            page_count=page_count,
            sha256=digest.hexdigest(),
        )
    except Exception:
        temporary_path.unlink(missing_ok=True)
        final_path.unlink(missing_ok=True)
        raise


def store_validated_pdf_upload(
    source_file: BinaryIO,
    *,
    content_type: str | None,
    storage_dir: str | Path,
    max_bytes: int,
    max_pages: int,
    max_extracted_characters: int = 5_000_000,
) -> StoredPdf:
    """Validate a browser upload and store it under a private random name."""
    normalized_content_type = (content_type or "").split(";", 1)[0].strip()
    if normalized_content_type not in {"application/pdf", "application/x-pdf"}:
        raise PdfSecurityError("Upload content type must be application/pdf.")

    return _store_pdf_stream(
        source_file,
        storage_dir=storage_dir,
        max_bytes=max_bytes,
        max_pages=max_pages,
        max_extracted_characters=max_extracted_characters,
    )


def store_validated_pdf(
    source_path: str | Path,
    *,
    allowed_import_roots: tuple[str, ...],
    storage_dir: str | Path,
    max_bytes: int,
    max_pages: int,
    max_extracted_characters: int = 5_000_000,
) -> StoredPdf:
    """Copy a validated PDF into managed storage under a random filename."""
    source = Path(source_path).expanduser().resolve()
    allowed_roots = [
        Path(root).expanduser().resolve()
        for root in allowed_import_roots
    ]

    if not allowed_roots:
        raise PdfSecurityError("Local PDF import is not configured.")

    if not any(source.is_relative_to(root) for root in allowed_roots):
        raise PdfSecurityError("PDF path is outside the allowed import roots.")

    if not source.is_file():
        raise PdfSecurityError("PDF file was not found.")

    with source.open("rb") as source_file:
        return _store_pdf_stream(
            source_file,
            storage_dir=storage_dir,
            max_bytes=max_bytes,
            max_pages=max_pages,
            max_extracted_characters=max_extracted_characters,
        )
