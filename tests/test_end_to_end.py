"""
End-to-end integration test creating a synthetic Sakthi Finance financial report PDF,
ingesting it, verifying chunks, embeddings, hybrid retrieval, and Qwen answer generation.
"""

from pathlib import Path
from decimal import Decimal
import fitz
import pytest

from ragcli.ingestion import IngestionEngine
from ragcli.retrieval import HybridRetriever
from ragcli.answer_generator import AnswerGenerator
from ragcli.repository import DocumentRepository
from ragcli.text_normalizer import parse_financial_number


@pytest.fixture(scope="module")
def sample_financial_pdf(tmp_path_factory) -> Path:
    """Generate a clean 2-page financial PDF with a Balance Sheet / P&L table."""
    tmp_dir = tmp_path_factory.mktemp("financial_pdf")
    pdf_path = tmp_dir / "sakthi_finance_sample_report.pdf"

    doc = fitz.open()

    # Page 1: Director's Report Narrative
    page1 = doc.new_page()
    text_p1 = (
        "Sakthi Finance Limited\n\n"
        "Directors' Report for Financial Year 2024-25\n\n"
        "The Directors have pleasure in presenting the Annual Report of Sakthi Finance Limited "
        "together with the audited financial statements for the year ended 31st March 2025 (FY 2024-25).\n\n"
        "Operations and Financial Highlights:\n"
        "During FY 2024-25, the company achieved total revenue of ₹ 25,430.50 Lakhs as compared to "
        "₹ 21,800.00 Lakhs in the previous financial year FY 2023-24. "
        "The company continues to maintain robust asset quality and capital adequacy.\n"
    )
    page1.insert_text((50, 72), text_p1, fontsize=11)

    # Page 2: Statement of Profit and Loss (Table)
    page2 = doc.new_page()
    text_p2 = (
        "Sakthi Finance Limited\n"
        "Statement of Profit and Loss for the year ended 31st March 2025\n"
        "(₹ in Lakhs)\n\n"
        "Particulars                      FY 2024-25      FY 2023-24\n"
        "-----------------------------------------------------------\n"
        "Revenue from Operations          24,500.50       20,950.00\n"
        "Other Income                        930.00          850.00\n"
        "Total Revenue (I + II)           25,430.50       21,800.00\n"
        "Finance Costs                    14,200.00       12,100.00\n"
        "Employee Benefits Expense         3,100.25        2,850.50\n"
        "Total Expenses                   17,300.25       14,950.50\n"
        "Profit Before Tax (PBT)           8,130.25        6,849.50\n"
        "Tax Expense                       2,045.00        1,720.00\n"
        "Profit After Tax (PAT)            6,085.25        5,129.50\n"
        "-----------------------------------------------------------\n"
        "Note 1: Refer to note on deferred tax liability.\n"
    )
    page2.insert_text((50, 72), text_p2, fontsize=10)

    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


def test_end_to_end_ingestion_and_retrieval(sample_financial_pdf):
    # 1. Ingestion
    engine = IngestionEngine()
    result = engine.ingest_pdf(sample_financial_pdf)

    assert result["status"] == "success"
    doc_id = result["document_id"]
    assert result["page_count"] == 2
    assert result["chunk_count"] > 0

    try:
        # 2. Hybrid Retrieval
        retriever = HybridRetriever()
        chunks = retriever.retrieve("Profit After Tax PAT FY 2024-25", top_k=25)

        assert len(chunks) > 0
        target_chunks = [c for c in chunks if c.document_id == doc_id]
        assert len(target_chunks) > 0
        found_target_chunk = any("6,085.25" in c.source_text or "Profit After Tax" in c.source_text for c in target_chunks)
        assert found_target_chunk

        # Verify parsed financial number
        dec, s = parse_financial_number("6,085.25")
        assert dec == Decimal("6085.25")

        # 3. Answer Generation with local Qwen
        generator = AnswerGenerator()
        ans = generator.generate_answer("What was the Profit After Tax (PAT) in FY 2024-25?", target_chunks)

        assert ans.confidence == "verified"
        assert "6,085.25" in ans.answer or "6085.25" in ans.answer
        assert len(ans.citations) > 0
        assert ans.citations[0].page_number in (1, 2)

    finally:
        # Clean up database
        repo = DocumentRepository()
        repo.delete_document(doc_id)
