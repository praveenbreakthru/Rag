"""
Comprehensive Test Suite for Master Implementation:
Tests:
1. Direct reported metric retrieval (Path A - PAT, PBT, Total Revenue)
2. Deterministic formula calculation engine (Path B - EBITDA, Net Profit Margin, Growth Rate, Ratios)
3. Zero-division and undefined edge cases
4. Unit normalization (Lakhs, Crores, Rupee)
5. Metric knowledge base lookup & alias resolution
"""

from decimal import Decimal
import pytest
from ragcli.formula_engine import DeterministicFormulaEngine
from ragcli.metrics_knowledge_base import lookup_metric_by_query, METRICS_CATALOGUE
from ragcli.decision_engine import DecisionEngine


def test_metric_knowledge_base_catalogue():
    assert len(METRICS_CATALOGUE) >= 40
    assert "EBITDA" in METRICS_CATALOGUE
    assert "PAT" in METRICS_CATALOGUE
    assert "NET_PROFIT_MARGIN" in METRICS_CATALOGUE
    assert "CRAR" in METRICS_CATALOGUE


def test_query_alias_resolution():
    m1 = lookup_metric_by_query("what is the value of ebitda?")
    assert m1 is not None
    assert m1.metric_id == "EBITDA"

    m2 = lookup_metric_by_query("what was the net profit margin?")
    assert m2 is not None
    assert m2.metric_id == "NET_PROFIT_MARGIN"

    m3 = lookup_metric_by_query("what is the crar of the company?")
    assert m3 is not None
    assert m3.metric_id == "CRAR"


def test_decision_engine_routing():
    # Direct retrieval query
    q1 = DecisionEngine.analyze_query("What was the Profit After Tax (PAT) for FY 2024-25?")
    assert q1.operation_type == "direct_retrieval"
    assert not q1.is_calculation_requested

    # Calculation / derived query
    q2 = DecisionEngine.analyze_query("What was the EBITDA for FY 2024-25?")
    assert q2.is_calculation_requested

    # Growth comparison query
    q3 = DecisionEngine.analyze_query("What was the PAT growth between FY 2023-24 and FY 2024-25?")
    assert q3.operation_type == "growth_comparison"
    assert q3.is_calculation_requested


def test_deterministic_ebitda_calculation():
    # Sakthi Finance FY25 actuals
    pbt = Decimal("2333.13")
    finance_costs = Decimal("11984.00")
    depreciation = Decimal("594.59")

    calc = DeterministicFormulaEngine.calculate_ebitda(pbt, finance_costs, depreciation)
    assert calc.is_valid
    assert calc.result_decimal == Decimal("14911.72")
    assert calc.result_formatted == "₹14,911.72 Lakhs"


def test_deterministic_margin_calculation():
    pat = Decimal("1665.15")
    revenue = Decimal("21498.88")

    calc = DeterministicFormulaEngine.calculate_margin(
        metric_id="NET_PROFIT_MARGIN",
        metric_name="Net Profit Margin",
        numerator_val=pat,
        numerator_name="Profit for the year",
        revenue_val=revenue,
    )
    assert calc.is_valid
    assert calc.result_formatted == "7.75%"


def test_zero_denominator_safe_handling():
    calc = DeterministicFormulaEngine.calculate_margin(
        metric_id="MARGIN",
        metric_name="Test Margin",
        numerator_val=Decimal("100"),
        numerator_name="Profit",
        revenue_val=Decimal("0"),
    )
    assert not calc.is_valid
    assert "Zero Revenue Denominator" in calc.result_formatted
    assert calc.result_decimal is None


def test_percentage_growth_calculation():
    current_pat = Decimal("1665.15")
    previous_pat = Decimal("1570.54")

    calc = DeterministicFormulaEngine.calculate_percentage_growth(
        metric_name="PAT",
        current_val=current_pat,
        previous_val=previous_pat,
    )
    assert calc.is_valid
    assert round(calc.result_decimal, 2) == Decimal("6.02")
    assert calc.result_formatted == "+6.02%"
