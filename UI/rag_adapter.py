"""
RAG Adapter for Sakthi Finance UI.
Connects the Streamlit UI to the existing HybridRetriever and AnswerGenerator
without modifying or duplicating any existing backend RAG logic.
"""

import sys
from pathlib import Path
from typing import Any, Optional

# Ensure project src/ directory is on Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from ragcli.retrieval import HybridRetriever
from ragcli.answer_generator import AnswerGenerator


class RAGAdapter:
    """Singleton adapter to reuse the existing RAG retrieval and generation pipeline."""
    _instance: Optional["RAGAdapter"] = None

    def __init__(self):
        self.retriever = HybridRetriever()
        self.generator = AnswerGenerator()

    @classmethod
    def get_instance(cls) -> "RAGAdapter":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def ask(
        self,
        question: str,
        conversation_history: Optional[list[dict[str, str]]] = None,
        financial_period: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Execute RAG retrieval and answer generation using the existing pipeline.
        Returns:
            dict containing:
                - answer (str)
                - citations (list of dicts)
                - latency_ms (float)
                - confidence (str)
        """
        # 1. Retrieve chunks using existing HybridRetriever
        chunks = self.retriever.retrieve(
            query=question,
            financial_period=financial_period,
        )

        # 2. Generate answer using existing AnswerGenerator
        res = self.generator.generate_answer(
            question=question,
            retrieved_chunks=chunks,
            conversation_history=conversation_history,
        )

        # 3. Format citations for UI presentation
        citations = [
            {
                "document_name": c.document_name,
                "page_number": c.page_number,
                "section_title": c.section_title or "-",
                "table_id": c.table_id or "-",
                "financial_period": c.financial_period or "-",
                "unit": c.unit or "-",
            }
            for c in res.citations
        ]

        return {
            "answer": res.answer,
            "citations": citations,
            "latency_ms": res.latency_ms,
            "confidence": res.confidence,
        }
