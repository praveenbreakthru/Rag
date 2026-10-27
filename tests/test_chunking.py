"""
Tests for financial table and narrative chunking behavior.
"""

from ragcli.schemas import ReconstructedTable
from ragcli.chunker import FinancialChunker


def test_table_chunking_repeats_headers():
    rows = [[f"Metric {i}", f"{100*i}", f"{120*i}"] for i in range(1, 51)]
    table = ReconstructedTable(
        table_id="tab_test_1",
        page_number=14,
        title="Statement of Profit and Loss",
        headers=["Particulars", "FY 2024-25", "FY 2023-24"],
        rows=rows,
        units="₹ in Lakhs",
        financial_periods=["FY 2024-25", "FY 2023-24"],
        footnotes=["Note 1: Refer accounting policies"],
    )

    chunker = FinancialChunker(max_table_rows=25)
    chunks = chunker.chunk_table(table)

    assert len(chunks) == 2
    for chunk in chunks:
        assert chunk.chunk_type == "table"
        assert "Statement of Profit and Loss" in chunk.source_text
        assert "₹ in Lakhs" in chunk.source_text
        assert "FY 2024-25" in chunk.source_text
        assert "Particulars" in chunk.source_text
        assert chunk.page_start == 14
        assert chunk.table_id == "tab_test_1"


def test_narrative_chunking_metadata():
    text = (
        "Sakthi Finance Limited presents its performance for FY 2024-25. "
        "The company recorded a substantial growth in assets under management (₹ in Lakhs). "
        "Profit after tax increased compared to the previous financial year."
    )
    chunker = FinancialChunker(chunk_size=500)
    chunks = chunker.chunk_narrative(text, page_number=5)

    assert len(chunks) == 1
    c = chunks[0]
    assert c.chunk_type == "narrative"
    assert c.page_start == 5
    assert "2024-25" in (c.financial_period or "")
    assert "Lakhs" in (c.unit or "")
