"""
Unicode-aware text normalization and financial number verification.
Preserves:
- Currency symbols: ₹, $, etc.
- Negative representations: (125.50), -125.50, - 125.50
- Percentages: 12.75%
- Comma separators: 1,602.62, 1,25,00,000 (Indian lakh/crore numbering)
- Dashes/em dashes: —, –, - (distinguishing placeholder dash from negative value)
- Financial periods: FY 2024-25, FY 2023-24, FY 2025-26
"""

import re
import unicodedata
from decimal import Decimal, InvalidOperation
from typing import Optional, Tuple


def normalize_unicode(text: str) -> str:
    """Normalize unicode characters while preserving financial symbols and dashes."""
    if not text:
        return ""
    # Normalize non-breaking spaces and exotic spaces to standard space
    text = re.sub(r"[\u00A0\u1680\u180e\u2000-\u200a\u202f\u205f\u3000]", " ", text)
    # Normalize special zero-width characters
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
    # Keep ₹, currency, en-dash, em-dash
    return text.strip()


def normalize_financial_notes(text: str) -> str:
    """
    Detect top-level note headings in the Notes section of financial statements
    (e.g., '4.\t CASH AND CASH EQUIVALENTS', '17.\t SUB-ORDINATED LIABILITIES')
    and normalize them into 'NOTE 4: CASH AND CASH EQUIVALENTS (4. CASH AND CASH EQUIVALENTS)'
    so that queries asking for 'Note 4', 'Notes 17', etc., match precisely via FTS, vector, and reranking.
    """
    if not text:
        return ""

    is_notes_page = (
        "notes forming an integral part" in text.lower()
        or "notes to the financial statements" in text.lower()
        or "notes to financial statements" in text.lower()
    )
    if not is_notes_page:
        return text

    lines = text.split("\n")
    new_lines = []
    # Primary financial note headings have a tab character after period (e.g. '4.\t CASH AND CASH EQUIVALENTS', '17.\t SUB-ORDINATED LIABILITIES')
    # whereas table data rows have regular spaces (e.g. '1. Agriculture and Allied Activities', '4. Personal Loans')
    note_pat = re.compile(r"^\s*(\d{1,2}[A-Za-z]?)\.\t\s*([A-Z0-9][A-Za-z0-9\s\(\),/–\-\'\"]{2,80})\s*$")

    for line in lines:
        m = note_pat.match(line)
        if m:
            num_str = m.group(1)
            title = m.group(2).strip()
            new_lines.append(f"NOTE {num_str}: {title} ({num_str}. {title})")
        else:
            new_lines.append(line)

    return "\n".join(new_lines)


def parse_financial_number(raw_val: str) -> Tuple[Optional[Decimal], Optional[str]]:
    """
    Parse a financial number from raw string.
    Returns (Decimal_value, cleaned_str).
    Preserves negative values indicated by parentheses or leading minus.
    Preserves distinction between '—' (empty/nil) and 0.0.
    """
    if not raw_val:
        return None, None

    val = raw_val.strip()

    # Distinguish empty or placeholder dash
    if val in ("—", "–", "-", "Nil", "NIL", "nil", "N.A.", "NA", ""):
        return None, "NIL"

    # Check for parentheses indicating negative: e.g. (250.50) or ( 250.50 )
    is_negative = False
    paren_match = re.match(r"^\s*\(\s*(.*?)\s*\)\s*$", val)
    if paren_match:
        is_negative = True
        val = paren_match.group(1).strip()
    elif val.startswith("-"):
        is_negative = True
        val = val[1:].strip()

    # Remove currency symbols and clean up
    val = re.sub(r"[₹$€£\s]", "", val)

    # Clean commas from numeric representation
    clean_numeric_str = val.replace(",", "")

    try:
        dec = Decimal(clean_numeric_str)
        if is_negative:
            dec = -dec
        return dec, str(dec)
    except (InvalidOperation, ValueError):
        return None, raw_val


def extract_financial_periods(text: str) -> list[str]:
    """Extract financial periods such as 'FY 2024-25', '2023-24', '2023 to 2024', '31st March 2025'."""
    patterns = [
        r"(?:FY|F\.Y\.)?\s*20\d{2}\s*[-–—]\s*(?:20)?\d{2}",
        r"(?:FY|F\.Y\.)?\s*20\d{2}\s+(?:to|and)\s+(?:20)?\d{2}",
        r"(?:As at|Year ended|Period ended)?\s*31(?:st)?\s+March,?\s+20\d{2}",
    ]
    periods = []
    for pat in patterns:
        matches = re.findall(pat, text, re.IGNORECASE)
        for m in matches:
            cleaned = re.sub(r"\s+", " ", m).strip()
            if cleaned not in periods:
                periods.append(cleaned)
    return periods


def extract_financial_unit(text: str) -> Optional[str]:
    """Detect stated financial units (e.g. (₹ in Lakhs), (Rs. in Crores))."""
    patterns = [
        r"\(\s*₹\s*in\s+(?:Lakhs?|Crores?|Thousands?)\s*\)",
        r"\(\s*(?:Rs\.|Rupees)\s*in\s+(?:Lakhs?|Crores?|Thousands?)\s*\)",
        r"(?:₹|Rs\.)\s*in\s+(?:Lakhs?|Crores?|Thousands?)",
    ]
    for pat in patterns:
        match = re.search(pat, text, re.IGNORECASE)
        if match:
            return match.group(0).strip()
    return None


# -------------------------------------------------------------------------
# Financial Metrics Taxonomy (For CFO/Executive Derived Metrics)
# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# Comprehensive Financial Metrics & Calculation Taxonomy
# -------------------------------------------------------------------------
FINANCIAL_METRICS_TAXONOMY: dict[str, dict] = {
    "ebitda": {
        "name": "EBITDA (Earnings Before Interest, Tax, Depreciation and Amortization)",
        "formula": "Profit before Tax + Finance Costs + Depreciation and Amortization Expense",
        "search_terms": [
            "Statement of Profit and Loss",
            "Profit before Tax",
            "Finance Costs",
            "Depreciation and Amortization Expense",
            "Revenue from Operations",
        ],
        "notes": "Derived from the Statement of Profit and Loss.",
    },
    "ebit": {
        "name": "EBIT (Operating Profit / Earnings Before Interest and Tax)",
        "formula": "Profit before Tax + Finance Costs",
        "search_terms": [
            "Statement of Profit and Loss",
            "Profit before Tax",
            "Finance Costs",
        ],
        "notes": "Operating earnings before financing charges and taxation.",
    },
    "pat": {
        "name": "Profit After Tax (PAT) / Profit for the year",
        "formula": "Profit before Tax - Tax Expense",
        "search_terms": [
            "Statement of Profit and Loss",
            "Profit for the year",
            "Tax Expense",
            "Current Tax",
        ],
        "notes": "Net profit earned by the company after all taxes.",
    },
    "net_profit_margin": {
        "name": "Net Profit Margin",
        "formula": "(Profit for the year / Total Revenue from Operations) * 100",
        "search_terms": [
            "Statement of Profit and Loss",
            "Profit for the year",
            "Total Revenue from operations",
        ],
        "notes": "Percentage of revenue converted into net profit.",
    },
    "operating_margin": {
        "name": "Operating Margin / EBITDA Margin",
        "formula": "(EBITDA / Total Revenue from Operations) * 100",
        "search_terms": [
            "Statement of Profit and Loss",
            "Total Revenue from operations",
            "Profit before Tax",
            "Finance Costs",
            "Depreciation and Amortization Expense",
        ],
        "notes": "Expressed as a percentage of revenue from operations.",
    },
    "interest_coverage_ratio": {
        "name": "Interest Coverage Ratio",
        "formula": "EBIT / Finance Costs (or (Profit before Tax + Finance Costs) / Finance Costs)",
        "search_terms": [
            "Statement of Profit and Loss",
            "Finance Costs",
            "Profit before Tax",
        ],
        "notes": "Measures company's debt servicing ability from operating profits.",
    },
    "net_worth": {
        "name": "Net Worth / Total Equity",
        "formula": "Equity Share Capital + Other Equity",
        "search_terms": [
            "Balance Sheet",
            "Equity Share Capital",
            "Other Equity",
            "Total Equity",
        ],
        "notes": "Total shareholders' equity from the Balance Sheet.",
    },
    "debt_to_equity": {
        "name": "Debt to Equity Ratio",
        "formula": "Total Debt (Borrowings + Debt Securities + Subordinated Liabilities) / Net Worth",
        "search_terms": [
            "Balance Sheet",
            "Debt Securities",
            "Borrowings",
            "Subordinated Liabilities",
            "Total Equity",
            "Other Equity",
        ],
        "notes": "Measures leverage and financial solvency.",
    },
    "total_debt": {
        "name": "Total Debt / Total Borrowings",
        "formula": "Debt Securities + Borrowings (other than debt securities) + Subordinated Liabilities",
        "search_terms": [
            "Balance Sheet",
            "Debt Securities",
            "Borrowings",
            "Subordinated Liabilities",
        ],
        "notes": "Total borrowing obligations from financial liabilities.",
    },
    "return_on_equity": {
        "name": "Return on Equity (ROE)",
        "formula": "(Profit for the year / Total Equity) * 100",
        "search_terms": [
            "Statement of Profit and Loss",
            "Profit for the year",
            "Balance Sheet",
            "Total Equity",
        ],
        "notes": "Measures profitability relative to shareholder equity.",
    },
    "return_on_assets": {
        "name": "Return on Assets (ROA)",
        "formula": "(Profit for the year / Total Assets) * 100",
        "search_terms": [
            "Statement of Profit and Loss",
            "Profit for the year",
            "Balance Sheet",
            "Total Assets",
        ],
        "notes": "Measures asset utilization efficiency.",
    },
    "capital_adequacy_ratio": {
        "name": "Capital to Risk-Weighted Assets Ratio (CRAR) / CAR",
        "formula": "(Tier I Capital + Tier II Capital) / Total Risk Weighted Assets",
        "search_terms": [
            "CRAR",
            "Capital Adequacy Ratio",
            "Tier I Capital",
            "Tier II Capital",
            "Risk Weighted Assets",
        ],
        "notes": "Mandatory NBFC regulatory solvency metric.",
    },
}


def get_metric_expansion(query: str) -> Optional[dict]:
    """Check if query is asking for a known financial metric or calculation and return search terms and formula."""
    q_lower = query.lower()
    for metric_key, info in FINANCIAL_METRICS_TAXONOMY.items():
        if metric_key in q_lower or metric_key.replace("_", " ") in q_lower:
            return info
    # Handle acronym matches and variations
    if re.search(r"\bebitda\b", q_lower):
        return FINANCIAL_METRICS_TAXONOMY["ebitda"]
    if re.search(r"\bebit\b", q_lower):
        return FINANCIAL_METRICS_TAXONOMY["ebit"]
    if re.search(r"\bpat\b|profit after tax", q_lower):
        return FINANCIAL_METRICS_TAXONOMY["pat"]
    if re.search(r"\boperating margin\b", q_lower):
        return FINANCIAL_METRICS_TAXONOMY["operating_margin"]
    if re.search(r"\bnet profit margin\b|\bnpm\b", q_lower):
        return FINANCIAL_METRICS_TAXONOMY["net_profit_margin"]
    if re.search(r"\binterest coverage\b", q_lower):
        return FINANCIAL_METRICS_TAXONOMY["interest_coverage_ratio"]
    if re.search(r"\bnet worth\b", q_lower):
        return FINANCIAL_METRICS_TAXONOMY["net_worth"]
    if re.search(r"\bdebt to equity\b|\bd/e\b", q_lower):
        return FINANCIAL_METRICS_TAXONOMY["debt_to_equity"]
    if re.search(r"\btotal debt\b|\btotal borrowings\b", q_lower):
        return FINANCIAL_METRICS_TAXONOMY["total_debt"]
    if re.search(r"\broe\b|return on equity", q_lower):
        return FINANCIAL_METRICS_TAXONOMY["return_on_equity"]
    if re.search(r"\broa\b|return on assets", q_lower):
        return FINANCIAL_METRICS_TAXONOMY["return_on_assets"]
    if re.search(r"\bcrar\b|capital adequacy", q_lower):
        return FINANCIAL_METRICS_TAXONOMY["capital_adequacy_ratio"]
    return None
