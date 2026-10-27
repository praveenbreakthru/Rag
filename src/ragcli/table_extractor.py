"""
Financial table extractor and reconstructor.
Extracts tables using PyMuPDF table finder, preserves headers, multi-period columns,
row labels, footnotes, and creates structured Markdown and JSON representations.
"""

import logging
from typing import Optional
import fitz
from tabulate import tabulate
from ragcli.schemas import ReconstructedTable
from ragcli.text_normalizer import (
    normalize_unicode,
    extract_financial_periods,
    extract_financial_unit,
)

logger = logging.getLogger(__name__)


class TableExtractor:
    def __init__(self, page: fitz.Page, page_number: int):
        self.page = page
        self.page_number = page_number

    def extract_tables(self) -> list[ReconstructedTable]:
        """Extract and reconstruct tables found on the page."""
        results: list[ReconstructedTable] = []
        # Try finding tables normally
        page_tabs = self.page.find_tables()
        tabs_to_process = list(page_tabs.tables) if page_tabs and page_tabs.tables else []

        # If no tables found, or if tables only have 2 columns with squashed text, check 90-degree rotation (landscape tables)
        is_squashed = False
        if tabs_to_process:
            first_ext = tabs_to_process[0].extract()
            if (
                first_ext
                and len(first_ext) > 1
                and len(first_ext[0]) == 2
                and len(str(first_ext[0][1] or "")) > 60
            ):
                is_squashed = True

        if not tabs_to_process or is_squashed:
            try:
                self.page.set_rotation(90)
                rot_tabs = self.page.find_tables()
                if rot_tabs and rot_tabs.tables:
                    tabs_to_process = list(rot_tabs.tables)
            finally:
                self.page.set_rotation(0)

        if not tabs_to_process:
            return results

        # Also get page text for header/unit detection context
        page_text = self.page.get_text("text")

        for idx, tab in enumerate(tabs_to_process):
            raw_data = tab.extract()
            if not raw_data or len(raw_data) < 2:
                continue

            # If raw_data has columns that contain newline-separated text, transpose column items to rows
            if len(raw_data) == 2 and any("\n" in str(c) for c in raw_data[1]):
                col_splits = [str(c or "").split("\n") for c in raw_data[1]]
                max_r = max(len(cs) for cs in col_splits)
                transposed = []
                for r_i in range(max_r):
                    transposed.append([col_splits[c_i][r_i] if r_i < len(col_splits[c_i]) else "" for c_i in range(len(raw_data[0]))])
                raw_data = [raw_data[0]] + transposed

            table_id = f"table_p{self.page_number}_{idx+1}"

            # Extract header and rows
            raw_headers = [normalize_unicode(str(c or "")) for c in raw_data[0]]
            raw_rows = []
            for r in raw_data[1:]:
                cleaned_row = [normalize_unicode(str(c or "")) for c in r]
                # Skip completely blank rows
                if any(cleaned_row):
                    raw_rows.append(cleaned_row)

            if not raw_rows:
                continue

            # Detect title from context preceding the table bbox
            bbox = tab.bbox  # (x0, y0, x1, y1)
            rect_above = fitz.Rect(bbox[0], max(0, bbox[1] - 60), bbox[2], bbox[1])
            text_above = self.page.get_text("text", clip=rect_above).strip()
            title = text_above.split("\n")[-1] if text_above else None

            # Detect unit from title or header context
            header_str = " ".join(raw_headers) + " " + (title or "")
            unit = extract_financial_unit(header_str) or extract_financial_unit(page_text)

            # Detect financial periods across headers
            periods = extract_financial_periods(header_str)

            # Detect footnotes beneath the table
            rect_below = fitz.Rect(bbox[0], bbox[3], bbox[2], min(self.page.rect.height, bbox[3] + 80))
            text_below = self.page.get_text("text", clip=rect_below).strip()
            footnotes = [line.strip() for line in text_below.split("\n") if line.strip().startswith(("*", "Note", "1.", "2."))]

            # Generate readable markdown representation
            markdown_repr = tabulate(raw_rows, headers=raw_headers, tablefmt="github")

            reconstructed = ReconstructedTable(
                table_id=table_id,
                page_number=self.page_number,
                title=title,
                headers=raw_headers,
                rows=raw_rows,
                units=unit,
                financial_periods=periods,
                footnotes=footnotes,
                markdown_repr=markdown_repr,
            )
            results.append(reconstructed)

        return results
