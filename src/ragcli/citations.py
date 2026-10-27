"""
Citation builder and verification module.
Maps retrieved evidence to clean source citations:
- Document filename
- Page number
- Table ID / Section title
- Stated financial year and unit
"""

from ragcli.schemas import DocumentChunk, Citation
from ragcli.database import get_db_cursor


def build_citations(chunks: list[DocumentChunk]) -> list[Citation]:
    """Extract unique, traceable citations from retrieved chunks."""
    citations: list[Citation] = []
    seen = set()

    # Get document filenames
    doc_ids = list({c.document_id for c in chunks if c.document_id})
    filename_map = {}
    if doc_ids:
        with get_db_cursor() as cur:
            cur.execute(
                "SELECT id, filename FROM documents WHERE id = ANY(%s::uuid[]);",
                (doc_ids,),
            )
            for row in cur.fetchall():
                filename_map[str(row[0])] = row[1]

    for chunk in chunks:
        doc_name = filename_map.get(chunk.document_id or "", "Document")
        key = (doc_name, chunk.page_start, chunk.section_title, chunk.table_id)
        if key in seen:
            continue
        seen.add(key)

        citations.append(
            Citation(
                document_name=doc_name,
                page_number=chunk.page_start,
                section_title=chunk.section_title,
                table_id=chunk.table_id,
                financial_period=chunk.financial_period,
                metric_name=chunk.metric_name,
                unit=chunk.unit,
                exact_source_text=chunk.source_text[:300] + ("..." if len(chunk.source_text) > 300 else ""),
            )
        )

    return citations
