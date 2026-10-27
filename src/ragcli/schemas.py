"""
Pydantic schemas and data structures for documents, pages, tables, chunks, and citations.
"""

from datetime import datetime
from typing import Any, Optional
from pydantic import BaseModel, Field


class DocumentPageRecord(BaseModel):
    page_number: int
    has_text: bool = True
    is_scanned: bool = False
    table_count: int = 0
    character_count: int = 0
    extraction_method: str = "native"  # 'native', 'layout', 'ocr'
    warnings: list[str] = Field(default_factory=list)


class DocumentRecord(BaseModel):
    id: Optional[str] = None
    filename: str
    file_hash: str
    page_count: int
    file_size_bytes: int
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[datetime] = None


class TableCell(BaseModel):
    row_idx: int
    col_idx: int
    value: str
    is_header: bool = False


class ReconstructedTable(BaseModel):
    table_id: str
    page_number: int
    page_end: Optional[int] = None
    title: Optional[str] = None
    headers: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)
    units: Optional[str] = None
    financial_periods: list[str] = Field(default_factory=list)
    footnotes: list[str] = Field(default_factory=list)
    markdown_repr: str = ""
    is_uncertain: bool = False


class DocumentChunk(BaseModel):
    id: Optional[str] = None
    document_id: Optional[str] = None
    chunk_index: int
    chunk_type: str  # 'narrative', 'table', 'footnote', 'header'
    page_start: int
    page_end: int
    section_title: Optional[str] = None
    table_id: Optional[str] = None
    financial_period: Optional[str] = None
    metric_name: Optional[str] = None
    unit: Optional[str] = None
    source_text: str
    normalized_text: str
    table_data: Optional[dict[str, Any]] = None
    similarity_score: Optional[float] = None
    rank_score: Optional[float] = None
    document_filename: Optional[str] = None


class Citation(BaseModel):
    document_name: str
    page_number: int
    section_title: Optional[str] = None
    table_id: Optional[str] = None
    financial_period: Optional[str] = None
    metric_name: Optional[str] = None
    reported_value: Optional[str] = None
    unit: Optional[str] = None
    exact_source_text: str


class AnswerResult(BaseModel):
    question: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    confidence: str = "verified"  # 'verified', 'partial', 'uncertain', 'unsupported'
    latency_ms: float = 0.0
