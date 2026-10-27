"""
Financial-aware chunking engine.
Separates narrative text, tables, and footnotes.
Preserves table headers, financial periods, and units across split table chunks.
"""

import re
from typing import Optional
from tabulate import tabulate
from ragcli.schemas import DocumentChunk, ReconstructedTable
from ragcli.config import settings
from ragcli.text_normalizer import extract_financial_periods, extract_financial_unit


class FinancialChunker:
    def __init__(
        self,
        chunk_size: int = settings.chunk_size,
        chunk_overlap: int = settings.chunk_overlap,
        max_table_rows: int = settings.table_max_rows_per_chunk,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_table_rows = max_table_rows

    def chunk_table(
        self,
        table: ReconstructedTable,
        document_id: Optional[str] = None,
        start_chunk_idx: int = 0,
    ) -> list[DocumentChunk]:
        """
        Split large tables into row-blocks while repeating:
        - Table title
        - Column headers
        - Stated units
        - Relevant periods
        """
        chunks: list[DocumentChunk] = []
        rows = table.rows
        headers = table.headers
        title = table.title or f"Table (Page {table.page_number})"
        unit_str = f" [Unit: {table.units}]" if table.units else ""
        period_str = f" [Period: {', '.join(table.financial_periods)}]" if table.financial_periods else ""

        if not rows:
            return chunks

        # Split into blocks of max_table_rows, ensuring chunk text doesn't exceed 2500 characters
        idx = 0
        current_chunk_idx = start_chunk_idx

        while idx < len(rows):
            # Take rows dynamically until character budget (2000 chars) or max_table_rows is reached
            step = 1
            while step <= self.max_table_rows and (idx + step) <= len(rows):
                candidate_slice = rows[idx : idx + step]
                test_table = tabulate(candidate_slice, headers=headers, tablefmt="github")
                if len(test_table) > 2200 and step > 1:
                    step -= 1
                    break
                step += 1

            step = max(1, step if (idx + step) <= len(rows) else len(rows) - idx)
            row_slice = rows[idx : idx + step]
            idx += step

            # Create markdown slice
            md_table = tabulate(row_slice, headers=headers, tablefmt="github")
            contextual_header = f"### {title}{unit_str}{period_str}\n"
            full_text = f"{contextual_header}\n{md_table}"

            if table.footnotes:
                fn_text = "\n\nFootnotes:\n" + "\n".join(f"- {fn}" for fn in table.footnotes)
                if len(full_text) + len(fn_text) <= 3000:
                    full_text += fn_text

            chunk = DocumentChunk(
                document_id=document_id,
                chunk_index=current_chunk_idx,
                chunk_type="table",
                page_start=table.page_number,
                page_end=table.page_end or table.page_number,
                section_title=title,
                table_id=table.table_id,
                financial_period=", ".join(table.financial_periods) if table.financial_periods else None,
                unit=table.units,
                source_text=full_text,
                normalized_text=full_text,
                table_data={
                    "headers": headers,
                    "rows": row_slice,
                    "unit": table.units,
                    "periods": table.financial_periods,
                },
            )
            chunks.append(chunk)
            current_chunk_idx += 1

        return chunks

    def chunk_narrative(
        self,
        text: str,
        page_number: int,
        document_id: Optional[str] = None,
        start_chunk_idx: int = 0,
        section_title: Optional[str] = None,
    ) -> list[DocumentChunk]:
        """
        Chunk narrative text along paragraph/sentence boundaries without splitting numbers or sentences mid-way.
        """
        chunks: list[DocumentChunk] = []
        if not text or not text.strip():
            return chunks

        # Split by double newline or paragraph markers
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        current_text = ""
        current_chunk_idx = start_chunk_idx

        for para in paragraphs:
            # If paragraph fits, append
            if len(current_text) + len(para) + 2 <= self.chunk_size:
                current_text = f"{current_text}\n\n{para}".strip()
            else:
                # Flush existing chunk
                if current_text:
                    periods = extract_financial_periods(current_text)
                    unit = extract_financial_unit(current_text)
                    chunks.append(
                        DocumentChunk(
                            document_id=document_id,
                            chunk_index=current_chunk_idx,
                            chunk_type="narrative",
                            page_start=page_number,
                            page_end=page_number,
                            section_title=section_title,
                            financial_period=", ".join(periods) if periods else None,
                            unit=unit,
                            source_text=current_text,
                            normalized_text=current_text,
                        )
                    )
                    current_chunk_idx += 1

                # If single paragraph exceeds chunk size, split by sentences
                if len(para) > self.chunk_size:
                    sentences = re.split(r"(?<=[.!?])\s+", para)
                    temp_sent = ""
                    for s in sentences:
                        if len(temp_sent) + len(s) + 1 <= self.chunk_size:
                            temp_sent = f"{temp_sent} {s}".strip()
                        else:
                            if temp_sent:
                                periods = extract_financial_periods(temp_sent)
                                unit = extract_financial_unit(temp_sent)
                                chunks.append(
                                    DocumentChunk(
                                        document_id=document_id,
                                        chunk_index=current_chunk_idx,
                                        chunk_type="narrative",
                                        page_start=page_number,
                                        page_end=page_number,
                                        section_title=section_title,
                                        financial_period=", ".join(periods) if periods else None,
                                        unit=unit,
                                        source_text=temp_sent,
                                        normalized_text=temp_sent,
                                    )
                                )
                                current_chunk_idx += 1
                            temp_sent = s
                    current_text = temp_sent
                else:
                    current_text = para

        # Flush trailing text
        if current_text:
            periods = extract_financial_periods(current_text)
            unit = extract_financial_unit(current_text)
            chunks.append(
                DocumentChunk(
                    document_id=document_id,
                    chunk_index=current_chunk_idx,
                    chunk_type="narrative",
                    page_start=page_number,
                    page_end=page_number,
                    section_title=section_title,
                    financial_period=", ".join(periods) if periods else None,
                    unit=unit,
                    source_text=current_text,
                    normalized_text=current_text,
                )
            )

        return chunks
