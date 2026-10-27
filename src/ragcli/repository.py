"""
Repository for PostgreSQL database interactions:
- Inserting and querying documents and pages
- Storing chunks and vector embeddings
- Deleting documents with cascade
- Duplicate document hash checks
"""

import json
import logging
from typing import Optional, Any
import psycopg
from ragcli.database import get_db_cursor
from ragcli.schemas import DocumentRecord, DocumentPageRecord, DocumentChunk

logger = logging.getLogger(__name__)


class DocumentRepository:
    def get_document_by_hash(self, file_hash: str) -> Optional[dict[str, Any]]:
        with get_db_cursor() as cur:
            cur.execute(
                "SELECT id, filename, file_hash, page_count, file_size_bytes FROM documents WHERE file_hash = %s",
                (file_hash,),
            )
            row = cur.fetchone()
            if row:
                return {
                    "id": str(row[0]),
                    "filename": row[1],
                    "file_hash": row[2],
                    "page_count": row[3],
                    "file_size_bytes": row[4],
                }
        return None

    def insert_document(
        self,
        doc_record: DocumentRecord,
        page_records: list[DocumentPageRecord],
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
        embedding_model: str,
        embedding_dim: int,
    ) -> str:
        """Insert document, pages, chunks, and embeddings atomically in one transaction."""
        with get_db_cursor() as cur:
            # 1. Insert document
            cur.execute(
                """
                INSERT INTO documents (filename, file_hash, page_count, file_size_bytes, metadata)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id;
                """,
                (
                    doc_record.filename,
                    doc_record.file_hash,
                    doc_record.page_count,
                    doc_record.file_size_bytes,
                    json.dumps(doc_record.metadata),
                ),
            )
            doc_id = str(cur.fetchone()[0])

            # 2. Insert pages
            for p in page_records:
                cur.execute(
                    """
                    INSERT INTO document_pages (
                        document_id, page_number, has_text, is_scanned, table_count, 
                        character_count, extraction_method, warnings
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
                    """,
                    (
                        doc_id,
                        p.page_number,
                        p.has_text,
                        p.is_scanned,
                        p.table_count,
                        p.character_count,
                        p.extraction_method,
                        json.dumps(p.warnings),
                    ),
                )

            # 3. Insert chunks & embeddings
            for idx, chunk in enumerate(chunks):
                cur.execute(
                    """
                    INSERT INTO document_chunks (
                        document_id, chunk_index, chunk_type, page_start, page_end,
                        section_title, table_id, financial_period, metric_name, unit,
                        source_text, normalized_text, table_data
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id;
                    """,
                    (
                        doc_id,
                        chunk.chunk_index,
                        chunk.chunk_type,
                        chunk.page_start,
                        chunk.page_end,
                        chunk.section_title,
                        chunk.table_id,
                        chunk.financial_period,
                        chunk.metric_name,
                        chunk.unit,
                        chunk.source_text,
                        chunk.normalized_text,
                        json.dumps(chunk.table_data) if chunk.table_data else None,
                    ),
                )
                chunk_id = str(cur.fetchone()[0])

                # Insert vector embedding
                emb = embeddings[idx]
                cur.execute(
                    """
                    INSERT INTO chunk_embeddings (chunk_id, embedding_model, embedding_dim, embedding)
                    VALUES (%s, %s, %s, %s);
                    """,
                    (chunk_id, embedding_model, embedding_dim, emb),
                )

        return doc_id

    def list_documents(self) -> list[dict[str, Any]]:
        with get_db_cursor() as cur:
            cur.execute(
                """
                SELECT d.id, d.filename, d.file_hash, d.page_count, d.file_size_bytes, 
                       d.created_at, count(c.id) as chunk_count
                FROM documents d
                LEFT JOIN document_chunks c ON d.id = c.document_id
                GROUP BY d.id
                ORDER BY d.created_at DESC;
                """
            )
            rows = cur.fetchall()
            return [
                {
                    "id": str(r[0]),
                    "filename": r[1],
                    "file_hash": r[2],
                    "page_count": r[3],
                    "file_size_bytes": r[4],
                    "created_at": r[5].isoformat() if r[5] else None,
                    "chunk_count": r[6],
                }
                for r in rows
            ]

    def delete_document(self, document_id: str) -> bool:
        with get_db_cursor() as cur:
            cur.execute("DELETE FROM documents WHERE id = %s RETURNING id;", (document_id,))
            return cur.fetchone() is not None

    def get_stats(self) -> dict[str, Any]:
        with get_db_cursor() as cur:
            cur.execute("SELECT count(*) FROM documents;")
            doc_count = cur.fetchone()[0]

            cur.execute("SELECT count(*) FROM document_pages;")
            page_count = cur.fetchone()[0]

            cur.execute("SELECT count(*), chunk_type FROM document_chunks GROUP BY chunk_type;")
            type_rows = cur.fetchall()
            chunk_types = {r[1]: r[0] for r in type_rows}

            cur.execute("SELECT count(*) FROM chunk_embeddings;")
            emb_count = cur.fetchone()[0]

            return {
                "total_documents": doc_count,
                "total_pages": page_count,
                "total_chunks": sum(chunk_types.values()),
                "chunk_types": chunk_types,
                "total_embeddings": emb_count,
            }
