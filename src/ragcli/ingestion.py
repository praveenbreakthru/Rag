"""
Ingestion pipeline:
- PDF inspection & validation
- Layout & table extraction
- Financial-aware chunking
- Embedding generation (Ollama)
- Atomic storage into PostgreSQL/pgvector
"""

import logging
from pathlib import Path
from typing import Any
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from ragcli.config import settings
from ragcli.schemas import DocumentRecord, DocumentPageRecord, DocumentChunk
from ragcli.pdf_extractor import PDFExtractor
from ragcli.chunker import FinancialChunker
from ragcli.ollama_client import OllamaClient
from ragcli.repository import DocumentRepository

logger = logging.getLogger(__name__)


class IngestionEngine:
    def __init__(self):
        self.repository = DocumentRepository()
        self.ollama = OllamaClient()
        self.chunker = FinancialChunker()

    def ingest_pdf(self, pdf_path: Path | str, console=None) -> dict[str, Any]:
        """Ingest a single PDF file with full financial chunking and pgvector embeddings."""
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        extractor = PDFExtractor(path)
        doc_record, extracted_pages = extractor.extract()

        # Check for duplicates by file hash
        existing = self.repository.get_document_by_hash(doc_record.file_hash)
        if existing:
            return {
                "status": "skipped",
                "reason": "duplicate",
                "document_id": existing["id"],
                "filename": doc_record.filename,
                "file_hash": doc_record.file_hash,
            }

        # Collect page records and assemble chunks
        page_records: list[DocumentPageRecord] = [ep.record for ep in extracted_pages]
        all_chunks: list[DocumentChunk] = []
        chunk_idx = 0

        for ep in extracted_pages:
            # 1. Process tables first
            for tab in ep.tables:
                tab_chunks = self.chunker.chunk_table(tab, start_chunk_idx=chunk_idx)
                for tc in tab_chunks:
                    all_chunks.append(tc)
                    chunk_idx += 1

            # 2. Process narrative text
            narrative_chunks = self.chunker.chunk_narrative(
                ep.narrative_text,
                page_number=ep.page_number,
                start_chunk_idx=chunk_idx,
            )
            for nc in narrative_chunks:
                all_chunks.append(nc)
                chunk_idx += 1

        if not all_chunks:
            raise ValueError(f"No extractable text or tables found in {path.name}")

        # Generate embeddings in batch via Ollama
        chunk_texts = [c.normalized_text for c in all_chunks]
        embeddings = []

        if console:
            console.print(f"[cyan]Generating embeddings for {len(chunk_texts)} chunks using {settings.embedding_model}...[/cyan]")

        for text in chunk_texts:
            emb = self.ollama.get_embedding(text)
            if len(emb) != settings.embedding_dim:
                # Fallback check or resize if model differs
                pass
            embeddings.append(emb)

        # Store in PostgreSQL
        doc_id = self.repository.insert_document(
            doc_record=doc_record,
            page_records=page_records,
            chunks=all_chunks,
            embeddings=embeddings,
            embedding_model=settings.embedding_model,
            embedding_dim=len(embeddings[0]) if embeddings else settings.embedding_dim,
        )

        return {
            "status": "success",
            "document_id": doc_id,
            "filename": doc_record.filename,
            "page_count": doc_record.page_count,
            "chunk_count": len(all_chunks),
            "table_chunk_count": sum(1 for c in all_chunks if c.chunk_type == "table"),
            "narrative_chunk_count": sum(1 for c in all_chunks if c.chunk_type == "narrative"),
        }
