"""
Deterministic Financial Calculation Engine.
Executes approved formulas with Decimal precision, checks denominators,
normalizes units (₹, ₹ Lakhs, ₹ Crores), and outputs auditable step-by-step traces.
"""

from decimal import Decimal, InvalidOperation, DivisionByZero
from typing import Optional, Any
from pydantic import BaseModel, Field

class CalculationResult(BaseModel):
    metric_id: str
    metric_name: str
    formula: str
    input_values: dict[str, Decimal] = Field(default_factory=dict)
    input_units: dict[str, str] = Field(default_factory=dict)
    input_sources: dict[str, str] = Field(default_factory=dict)
    result_decimal: Optional[Decimal] = None
    result_formatted: str
    unit: str
    step_by_step_trace: list[str] = Field(default_factory=list)
    is_valid: bool = True
    error_message: Optional[str] = None


class DeterministicFormulaEngine:
    """Executes validated deterministic calculations for financial metrics."""

    @staticmethod
    def normalize_to_lakhs(val: Decimal, unit_str: str) -> Decimal:
        """Normalize amounts to ₹ Lakhs for arithmetic consistency."""
        u = unit_str.lower()
        if "crore" in u:
            return val * Decimal("100")
        elif "lakh" in u:
            return val
        elif "thousand" in u:
            return val / Decimal("100")
        elif "rupee" in u or "₹" in u:
            return val / Decimal("100000")
        return val

    @classmethod
    def calculate_ebitda(
        cls,
        pbt: Decimal,
        finance_costs: Decimal,
        depreciation_amortization: Decimal,
        period: str = "FY 2024-25",
    ) -> CalculationResult:
        formula = "PBT + Finance Costs + Depreciation & Amortization"
        result = pbt + finance_costs + depreciation_amortization
        return CalculationResult(
            metric_id="EBITDA",
            metric_name="EBITDA",
            formula=formula,
            input_values={"PBT": pbt, "Finance Costs": finance_costs, "Depreciation": depreciation_amortization},
            input_units={"PBT": "₹ Lakhs", "Finance Costs": "₹ Lakhs", "Depreciation": "₹ Lakhs"},
            result_decimal=result,
            result_formatted=f"₹{result:,.2f} Lakhs",
            unit="₹ Lakhs",
            step_by_step_trace=[
                f"Formula: {formula}",
                f"Inputs: PBT = ₹{pbt:,.2f} Lakhs, Finance Costs = ₹{finance_costs:,.2f} Lakhs, Depreciation = ₹{depreciation_amortization:,.2f} Lakhs",
                f"Calculation: ₹{pbt:,.2f} + ₹{finance_costs:,.2f} + ₹{depreciation_amortization:,.2f} = ₹{result:,.2f} Lakhs",
            ],
            is_valid=True,
        )

    @classmethod
    def calculate_margin(
        cls,
        metric_id: str,
        metric_name: str,
        numerator_val: Decimal,
        numerator_name: str,
        revenue_val: Decimal,
        revenue_name: str = "Total Revenue from operations",
    ) -> CalculationResult:
        formula = f"({numerator_name} / {revenue_name}) * 100"
        if revenue_val == Decimal("0"):
            return CalculationResult(
                metric_id=metric_id,
                metric_name=metric_name,
                formula=formula,
                result_formatted="Undefined (Zero Revenue Denominator)",
                unit="%",
                is_valid=False,
                error_message="Cannot calculate margin: Revenue denominator is zero.",
            )

        margin = (numerator_val / revenue_val) * Decimal("100")
        return CalculationResult(
            metric_id=metric_id,
            metric_name=metric_name,
            formula=formula,
            input_values={numerator_name: numerator_val, revenue_name: revenue_val},
            result_decimal=margin,
            result_formatted=f"{margin:.2f}%",
            unit="%",
            step_by_step_trace=[
                f"Formula: {formula}",
                f"Inputs: {numerator_name} = ₹{numerator_val:,.2f} Lakhs, {revenue_name} = ₹{revenue_val:,.2f} Lakhs",
                f"Calculation: (₹{numerator_val:,.2f} / ₹{revenue_val:,.2f}) * 100 = {margin:.2f}%",
            ],
            is_valid=True,
        )

    @classmethod
    def calculate_percentage_growth(
        cls,
        metric_name: str,
        current_val: Decimal,
        previous_val: Decimal,
        current_period: str = "FY 2024-25",
        previous_period: str = "FY 2023-24",
    ) -> CalculationResult:
        formula = "((Current Value - Previous Value) / |Previous Value|) * 100"
        if previous_val == Decimal("0"):
            return CalculationResult(
                metric_id="GROWTH_RATE",
                metric_name=f"{metric_name} Growth (%)",
                formula=formula,
                result_formatted="Undefined (Prior Period Value is Zero)",
                unit="%",
                is_valid=False,
                error_message="Prior period value is zero; conventional percentage growth is undefined.",
            )

        diff = current_val - previous_val
        growth = (diff / abs(previous_val)) * Decimal("100")
        direction = "increase" if diff >= 0 else "decrease"

        return CalculationResult(
            metric_id="GROWTH_RATE",
            metric_name=f"{metric_name} Growth (%)",
            formula=formula,
            input_values={f"{metric_name} ({current_period})": current_val, f"{metric_name} ({previous_period})": previous_val},
            result_decimal=growth,
            result_formatted=f"{growth:+.2f}%",
            unit="%",
            step_by_step_trace=[
                f"Formula: {formula}",
                f"{current_period}: ₹{current_val:,.2f} Lakhs",
                f"{previous_period}: ₹{previous_val:,.2f} Lakhs",
                f"Absolute Difference: {diff:+,.2f} Lakhs ({direction})",
                f"Percentage Change: ({diff:,.2f} / {abs(previous_val):,.2f}) * 100 = {growth:+.2f}%",
            ],
            is_valid=True,
        )

    @classmethod
    def calculate_ratio(
        cls,
        metric_id: str,
        metric_name: str,
        numerator: Decimal,
        numerator_name: str,
        denominator: Decimal,
        denominator_name: str,
        as_percentage: bool = False,
    ) -> CalculationResult:
        formula = f"{numerator_name} / {denominator_name}"
        if denominator == Decimal("0"):
            return CalculationResult(
                metric_id=metric_id,
                metric_name=metric_name,
                formula=formula,
                result_formatted="Undefined (Zero Denominator)",
                unit="ratio",
                is_valid=False,
                error_message=f"Cannot compute ratio: {denominator_name} is zero.",
            )

        val = numerator / denominator
        if as_percentage:
            val = val * Decimal("100")
            formatted = f"{val:.2f}%"
            unit = "%"
        else:
            formatted = f"{val:.2f}x"
            unit = "ratio"

        return CalculationResult(
            metric_id=metric_id,
            metric_name=metric_name,
            formula=formula,
            input_values={numerator_name: numerator, denominator_name: denominator},
            result_decimal=val,
            result_formatted=formatted,
            unit=unit,
            step_by_step_trace=[
                f"Formula: {formula}",
                f"Inputs: {numerator_name} = {numerator:,.2f}, {denominator_name} = {denominator:,.2f}",
                f"Result: {formatted}",
            ],
            is_valid=True,
        )

    @classmethod
    def verify_and_correct_arithmetic(cls, answer_text: str) -> str:
        """
        Audit arithmetic addition expressions in LLM-generated answers.
        Detects when the LLM correctly extracts the audited line items (e.g. 1,735.97 + 10,926.13 + 546.93)
        but makes a token-level math addition mistake (e.g. outputs 12,209.03 instead of 13,209.03).
        Validates the math deterministically and replaces incorrect sums with the exact Decimal result.
        """
        import re
        pattern = r"(\d[\d,]*\.\d{2}(?:\s*\+\s*\d[\d,]*\.\d{2})+)"
        for m in re.finditer(pattern, answer_text):
            expr = m.group(1)
            try:
                nums = [Decimal(n.strip().replace(",", "")) for n in expr.split("+")]
                expected_sum = sum(nums)
                expected_str = f"{expected_sum:,.2f}"

                end_pos = m.end()
                # Search for candidate sum numbers following the addition expression
                following_text = answer_text[end_pos:end_pos + 400]
                next_nums = re.findall(r"(\d[\d,]*\.\d{2})", following_text)
                for n_str in next_nums:
                    try:
                        cand_dec = Decimal(n_str.replace(",", ""))
                        if cand_dec != expected_sum and abs(cand_dec - expected_sum) in (
                            Decimal("1000"), Decimal("2000"), Decimal("100"), Decimal("200"),
                            Decimal("10"), Decimal("1"), Decimal("0.01"), Decimal("0.10")
                        ):
                            # Replace the erroneous sum with the mathematically verified sum
                            answer_text = answer_text[:end_pos] + answer_text[end_pos:].replace(n_str, expected_str)
                    except Exception:
                        continue
            except Exception:
                continue
        return answer_text
