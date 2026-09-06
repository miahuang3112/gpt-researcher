from io import BytesIO
from pathlib import Path

import fitz
import pytest

from backend.academic_api.services.secure_pdf import (
    PdfSecurityError,
    store_validated_pdf,
    store_validated_pdf_upload,
)


def create_pdf(path: Path, *, pages: int = 1) -> None:
    document = fitz.open()
    for page_number in range(pages):
        page = document.new_page()
        page.insert_text((72, 72), f"Safe academic page {page_number + 1}")
    document.save(str(path))
    document.close()


def store(source: Path, storage: Path, **limits):
    return store_validated_pdf(
        source,
        allowed_import_roots=(str(source.parent),),
        storage_dir=storage,
        max_bytes=limits.get("max_bytes", 1024 * 1024),
        max_pages=limits.get("max_pages", 10),
    )


def test_valid_pdf_is_copied_to_random_private_storage(tmp_path: Path) -> None:
    source = tmp_path / "user-controlled-name.pdf"
    storage = tmp_path / "managed"
    create_pdf(source, pages=2)

    first = store(source, storage)
    second = store(source, storage)

    assert first.path != second.path
    assert first.path.name != source.name
    assert first.path.suffix == ".pdf"
    assert first.path.read_bytes() == source.read_bytes()
    assert first.page_count == 2
    assert len(first.sha256) == 64
    assert first.path.stat().st_mode & 0o077 == 0


def test_rejects_path_outside_allowed_import_root(tmp_path: Path) -> None:
    source = tmp_path / "paper.pdf"
    create_pdf(source)

    with pytest.raises(PdfSecurityError, match="outside"):
        store_validated_pdf(
            source,
            allowed_import_roots=(str(tmp_path / "other"),),
            storage_dir=tmp_path / "managed",
            max_bytes=1024 * 1024,
            max_pages=10,
        )


def test_rejects_fake_pdf_signature_and_removes_partial_file(tmp_path: Path) -> None:
    source = tmp_path / "fake.pdf"
    storage = tmp_path / "managed"
    source.write_bytes(b"not really a pdf")

    with pytest.raises(PdfSecurityError, match="signature"):
        store(source, storage)

    assert not storage.exists() or list(storage.iterdir()) == []


def test_rejects_size_and_page_limits(tmp_path: Path) -> None:
    source = tmp_path / "large.pdf"
    storage = tmp_path / "managed"
    create_pdf(source, pages=3)

    with pytest.raises(PdfSecurityError, match="size limit"):
        store(source, storage, max_bytes=10)

    with pytest.raises(PdfSecurityError, match="page limit"):
        store(source, storage, max_pages=2)

    assert list(storage.iterdir()) == []


def test_rejects_malformed_and_encrypted_pdf(tmp_path: Path) -> None:
    malformed = tmp_path / "malformed.pdf"
    malformed.write_bytes(b"%PDF-1.7\ninvalid")

    with pytest.raises(PdfSecurityError, match="malformed"):
        store(malformed, tmp_path / "managed")

    encrypted = tmp_path / "encrypted.pdf"
    document = fitz.open()
    document.new_page()
    document.save(
        str(encrypted),
        encryption=fitz.PDF_ENCRYPT_AES_256,
        owner_pw="owner",
        user_pw="reader",
    )
    document.close()

    with pytest.raises(PdfSecurityError, match="Encrypted"):
        store(encrypted, tmp_path / "managed")


def test_rejects_pdf_with_embedded_payload(tmp_path: Path) -> None:
    source = tmp_path / "attachment.pdf"
    document = fitz.open()
    document.new_page()
    document.embfile_add("payload.txt", b"untrusted executable content")
    document.save(str(source))
    document.close()

    with pytest.raises(PdfSecurityError, match="active content"):
        store(source, tmp_path / "managed")


def test_browser_upload_is_streamed_to_private_random_storage(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.pdf"
    storage = tmp_path / "managed"
    create_pdf(source, pages=2)

    stored = store_validated_pdf_upload(
        BytesIO(source.read_bytes()),
        content_type="application/pdf",
        storage_dir=storage,
        max_bytes=1024 * 1024,
        max_pages=10,
    )

    assert stored.path.parent == storage.resolve()
    assert stored.path.name != source.name
    assert stored.page_count == 2
    assert stored.path.stat().st_mode & 0o077 == 0


def test_browser_upload_rejects_wrong_type_and_oversize_without_residue(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.pdf"
    storage = tmp_path / "managed"
    create_pdf(source)
    content = source.read_bytes()

    with pytest.raises(PdfSecurityError, match="content type"):
        store_validated_pdf_upload(
            BytesIO(content),
            content_type="text/plain",
            storage_dir=storage,
            max_bytes=1024 * 1024,
            max_pages=10,
        )

    with pytest.raises(PdfSecurityError, match="size limit"):
        store_validated_pdf_upload(
            BytesIO(content),
            content_type="application/pdf; charset=binary",
            storage_dir=storage,
            max_bytes=10,
            max_pages=10,
        )

    assert not storage.exists() or list(storage.iterdir()) == []
