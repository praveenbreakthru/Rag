"""
PDF extraction engine.
Combines page-level text extraction with layout-aware table extraction,
preserving page numbers, reading order, and table boundaries.
"""

from pathlib import Path
from typing import Generator
import fitz
from ragcli.schemas import DocumentRecord, DocumentPageRecord, ReconstructedTable
from ragcli.pdf_inspector import PDFInspector
from ragcli.table_extractor import TableExtractor
from ragcli.text_normalizer import normalize_unicode, normalize_financial_notes


class ExtractedPage:
    def __init__(
        self,
        page_number: int,
        narrative_text: str,
        tables: list[ReconstructedTable],
        record: DocumentPageRecord,
    ):
        self.page_number = page_number
        self.narrative_text = narrative_text
        self.tables = tables
        self.record = record


class PDFExtractor:
    def __init__(self, file_path: Path | str):
        self.file_path = Path(file_path)
        self.inspector = PDFInspector(self.file_path)

    def extract(self) -> tuple[DocumentRecord, list[ExtractedPage]]:
        """Extract pages and tables from PDF."""
        doc_record, page_records = self.inspector.inspect()
        doc = fitz.open(self.file_path)
        extracted_pages: list[ExtractedPage] = []

        try:
            for page_record in page_records:
                page_idx = page_record.page_number - 1
                page = doc[page_idx]

                # 1. Extract tables
                tab_extractor = TableExtractor(page, page_record.page_number)
                tables = tab_extractor.extract_tables()

                # 2. Extract text blocks
                # Check if page has sideways / landscape text flow (where text blocks have height >> width)
                # Require at least 5 meaningful tall blocks (with text length > 10) to avoid false positives on decorative glyphs
                blocks = page.get_text("blocks")
                body_blocks = [b for b in blocks if b[1] < 730]
                tall_meaningful_blocks = [
                    b for b in body_blocks 
                    if (b[3] - b[1]) / max(1.0, (b[2] - b[0])) > 4.0 and len(b[4].strip()) > 10
                ]
                is_landscape_text = len(tall_meaningful_blocks) >= 5

                if is_landscape_text:
                    # Filter out header/footer page margins and sort along landscape reading flow:
                    # Lines advance with increasing X (banded by ~12pt), reading horizontally with decreasing Y
                    words = page.get_text("words")
                    content_words = [w for w in words if w[1] <= 730 and w[0] >= 35]
                    sorted_words = sorted(content_words, key=lambda w: (round(w[0] / 12), -w[1]))
                    lines = []
                    current_band = None
                    cur_line = []
                    for w in sorted_words:
                        band = round(w[0] / 12)
                        if band != current_band:
                            if cur_line:
                                lines.append(" ".join(cur_line))
                            current_band = band
                            cur_line = [w[4]]
                        else:
                            cur_line.append(w[4])
                    if cur_line:
                        lines.append(" ".join(cur_line))
                    raw_text = "\n".join(lines)
                else:
                    raw_text = page.get_text("text")

                normalized_text = normalize_unicode(raw_text)
                normalized_text = normalize_financial_notes(normalized_text)

                extracted_pages.append(
                    ExtractedPage(
                        page_number=page_record.page_number,
                        narrative_text=normalized_text,
                        tables=tables,
                        record=page_record,
                    )
                )

            return doc_record, extracted_pages
        finally:
            doc.close()
