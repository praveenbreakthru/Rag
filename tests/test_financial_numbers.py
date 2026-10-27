"""
Unit tests for financial number parsing and unicode normalization.
"""

from decimal import Decimal
from ragcli.text_normalizer import (
    normalize_unicode,
    parse_financial_number,
    extract_financial_periods,
    extract_financial_unit,
)


def test_financial_number_parsing():
    # 1. Standard comma separated
    dec, s = parse_financial_number("1,602.62")
    assert dec == Decimal("1602.62")

    # 2. Parentheses indicating negative
    dec, s = parse_financial_number("(250.50)")
    assert dec == Decimal("-250.50")

    # 3. Currency symbol with Indian numbering
    dec, s = parse_financial_number("₹ 1,25,000.00")
    assert dec == Decimal("125000.00")

    # 4. Leading negative
    dec, s = parse_financial_number("-125.40")
    assert dec == Decimal("-125.40")

    # 5. Crores style grouping
    dec, s = parse_financial_number("1,25,00,000")
    assert dec == Decimal("12500000")

    # 6. Distinguish placeholder dash from zero
    dec, s = parse_financial_number("—")
    assert dec is None
    assert s == "NIL"

    dec, s = parse_financial_number("-")
    assert dec is None
    assert s == "NIL"


def test_unicode_normalization():
    raw = "Sakthi\u00a0Finance\u200b Limited"
    clean = normalize_unicode(raw)
    assert clean == "Sakthi Finance Limited"


def test_period_and_unit_extraction():
    sample = "Statement of Profit and Loss for the year ended 31st March 2025 (₹ in Lakhs) FY 2024-25"
    periods = extract_financial_periods(sample)
    assert any("2024-25" in p for p in periods)
    assert any("31st March 2025" in p or "31" in p for p in periods)

    unit = extract_financial_unit(sample)
    assert unit is not None
    assert "Lakhs" in unit
