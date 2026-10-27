"""
Retrieval-First Decision and Query Understanding Engine.
Classifies CFO queries into:
- Path A: Direct Retrieval (Metric is explicitly reported in Annual Report)
- Path B: Formula-Based Calculation (Metric is derived or requested arithmetic)
- Path C: Reconciliation (Both reported and calculated values exist)
"""

import re
from typing import Optional, Tuple
from pydantic import BaseModel, Field

from ragcli.metrics_knowledge_base import lookup_metric_by_query, MetricDefinition
from ragcli.formula_engine import DeterministicFormulaEngine, CalculationResult

class QueryIntent(BaseModel):
    original_query: str
    target_metric: Optional[MetricDefinition] = None
    operation_type: str = "direct_retrieval"  # "direct_retrieval", "calculation", "growth_comparison", "ratio", "sum"
    financial_period: str = "FY 2024-25"
    comparison_period: Optional[str] = None
    is_calculation_requested: bool = False
    requires_reconciliation: bool = False


class DecisionEngine:
    """Orchestrates query classification, retrieval-first routing, and formula calculation dispatch."""

    @classmethod
    def analyze_query(cls, query: str) -> QueryIntent:
        q_lower = query.lower()

        # Check for growth, comparison, or difference keywords
        is_growth = any(k in q_lower for k in ("growth", "percentage change", "change in", "increased", "decreased", "difference between", "compared with"))
        is_ratio = any(k in q_lower for k in ("ratio", "margin", "coverage"))
        # An arithmetic sum is only requested if user explicitly asks to sum or add across years, not when asking for "total value of [metric] of year 1 and year 2"
        is_sum = any(k in q_lower for k in ("sum of", "aggregate of", "calculate sum", "add together")) or (
            "total of" in q_lower and not any(m in q_lower for m in ("profit", "asset", "revenue", "income", "expense", "tax", "ebitda"))
        )
        is_calc_keyword = any(k in q_lower for k in ("calculate", "compute", "derive", "what is the value of ebitda"))

        # Detect period
        period = "FY 2024-25"
        comp_period = None
        if "2023-24" in q_lower or "2024" in q_lower and "2025" in q_lower:
            comp_period = "FY 2023-24"

        metric = lookup_metric_by_query(query)

        # Decide operation
        if is_growth:
            op_type = "growth_comparison"
        elif is_sum:
            op_type = "sum"
        elif metric and (metric.calculation_method == "formula_only" or (metric.is_nbfc_specific and metric.metric_id == "EBITDA")):
            op_type = "calculation"
        elif is_ratio or is_calc_keyword:
            op_type = "calculation" if (metric and metric.calculation_method != "direct_only") else "direct_retrieval"
        else:
            op_type = "direct_retrieval"

        return QueryIntent(
            original_query=query,
            target_metric=metric,
            operation_type=op_type,
            financial_period=period,
            comparison_period=comp_period,
            is_calculation_requested=(op_type != "direct_retrieval"),
        )
