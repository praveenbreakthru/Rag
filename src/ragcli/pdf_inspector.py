"""
PDF inspection and quality validation module.
Inspects page count, extractable text, scans vs text, table indicators, and file hashes.
"""

import hashlib
import logging
from pathlib import Path
import fitz  # PyMuPDF
from ragcli.schemas import DocumentRecord, DocumentPageRecord

logger = logging.getLogger(__name__)


def compute_file_hash(file_path: Path | str) -> str:
    """Calculate SHA-256 hash of a file for duplicate detection."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()


class PDFInspector:
    def __init__(self, file_path: Path | str):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"PDF file does not exist: {self.file_path}")

    def inspect(self) -> tuple[DocumentRecord, list[DocumentPageRecord]]:
        """Perform non-destructive inspection of PDF."""
        file_hash = compute_file_hash(self.file_path)
        file_size = self.file_path.stat().st_size

        doc = fitz.open(self.file_path)
        page_records: list[DocumentPageRecord] = []

        try:
            for page_idx in range(len(doc)):
                page = doc[page_idx]
                page_num = page_idx + 1

                # Extract raw text
                text = page.get_text("text").strip()
                char_count = len(text)
                has_text = char_count > 30

                # Detect if page is likely scanned (images present with very low text count)
                images = page.get_images()
                is_scanned = len(images) > 0 and char_count < 50

                # Detect table presence (using PyMuPDF find_tables)
                tabs = page.find_tables()
                table_count = len(tabs.tables) if tabs else 0

                warnings = []
                if is_scanned:
                    warnings.append("Page appears to be a scanned image or contains sparse text.")
                if char_count == 0:
                    warnings.append("Page has 0 extractable characters.")

                page_records.append(
                    DocumentPageRecord(
                        page_number=page_num,
                        has_text=has_text,
                        is_scanned=is_scanned,
                        table_count=table_count,
                        character_count=char_count,
                        extraction_method="native" if not is_scanned else "ocr_required",
                        warnings=warnings,
                    )
                )

            doc_record = DocumentRecord(
                filename=self.file_path.name,
                file_hash=file_hash,
                page_count=len(doc),
                file_size_bytes=file_size,
                metadata={
                    "format": doc.metadata.get("format"),
                    "title": doc.metadata.get("title"),
                    "author": doc.metadata.get("author"),
                },
            )

            return doc_record, page_records
        finally:
            doc.close()
