"""
Source-grounded financial answer generator using local Qwen 2.5 7B via Ollama.
Ensures:
- Direct answer first
- Financial period, units, and figures reported exactly as in source
- Never invent figures or calculate from memory
- Clear citation attribution
"""

import time
import logging
from decimal import Decimal
from typing import Optional
from ragcli.ollama_client import OllamaClient
from ragcli.schemas import DocumentChunk, AnswerResult
from ragcli.citations import build_citations
from ragcli.formula_engine import DeterministicFormulaEngine

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert Financial Analyst and Document Auditor for Sakthi Finance.
Your task is to answer user questions using ONLY the provided verified source evidence.

MANDATORY AUDIT RULES:
1. RETRIEVAL-FIRST MANDATE:
   - Always check whether the requested financial metric is explicitly reported in the source documents first.
   - When asked for "profit of the year", "profit for the year", or "profit", extract the exact printed line item titled "Profit for the year" or "Profit after tax for the year" from the Statement of Profit and Loss for the requested financial period. You may also report "Profit before Tax" if clarifying both pre-tax and post-tax.
   - CRITICAL: DO NOT report "Profit available for Appropriation" or "Balance carried forward to Balance Sheet" (which include multi-year retained earnings). DO NOT attempt manual calculations or formulas when the line item "Profit for the year" is printed directly in the statement table.
   - If the reported metric (e.g. Profit for the year, Profit after tax, Profit before Tax, Total Revenue from operations, Total Assets, Net Worth, Cash and Cash Equivalents) is explicitly available as a line item in the retrieved evidence, REPORT THE EXACT REPORTED VALUE DIRECTLY.
   - Each user question must be answered using the current retrieved evidence for that specific question and period; do not carry over previous years or assumptions from earlier turns unless explicitly asked.

2. SCHEDULES & PROPERTY, PLANT AND EQUIPMENT:
   - Look at the primary schedule table for the requested metric:
     * Carrying amount (Gross): report the exact total row for the period.
     * Net Carrying amount (Net): report the exact total row for the period.
   - If there is a separate subsidiary note below the main table for assets pledged against borrowings or debt securities, clearly distinguish between total company carrying amount and pledged carrying value.

3. BALANCE SHEET STATEMENT AUDITING:
   - When asked for details of the balance sheet, total assets, or liabilities and equity:
     * Locate the primary Balance Sheet statement in the retrieved evidence.
     * Extract and present the exact reported line items and totals:
       - Financial Assets and Non-Financial Assets, totaling to Total Assets.
       - Financial Liabilities and Non-Financial Liabilities.
       - Equity (Equity Share Capital, Other Equity) and Total Liabilities and Equity.
     * Present the exact figures as reported in the evidence for all requested comparative periods.

4. STATEMENT LINE ITEMS & UNITS:
   - Report financial figures with the exact currency, unit (e.g., ` Lakhs / ₹ Lakhs), and date exactly as printed in the source document header.
   - Retain the exact printed unit from the source statement. Never convert or change units unless explicitly asked.

5. CASH FLOW STATEMENT AUDITING:
   - The Statement of Cash Flows consists of Operating, Investing, and Financing activities.
   - When asked for a specific line item in the Cash Flow Statement:
     * Locate the exact row in the corresponding activity section.
     * Extract the exact value for each requested year from the comparative columns.
     * Do not confuse supplementary memo disclosures (e.g., interest received/paid) with the main cash flow statement subtotals.

6. NOTES, SCHEDULES & MULTI-PERIOD TABLES (CLEAN MARKDOWN TABLE MANDATE):
   - When the user asks for "Notes", "Note details", branch directories, or multi-year financial items:
     * Format the output as a clean, structured Markdown table with proper column headers and rows.
     * Never output compressed or malformed pipes. Ensure table columns and rows are cleanly aligned for readability.

7. FINANCIAL CALCULATIONS & DERIVED METRICS:
   - This rule applies ONLY to derived calculations (such as EBITDA, Ratios, or Growth percentages) when not explicitly reported as a line item.
   - When calculating derived metrics such as EBITDA:
     * State the formula: EBITDA = Profit before Tax + Finance Costs + Depreciation and Amortization Expense.
     * Extract each component directly from the Statement of Profit and Loss for that financial period.
     * Perform the addition dynamically using the extracted components and show the step-by-step arithmetic.
     * Never calculate profit manually when the printed "Profit for the year" and "Profit before Tax" line items exist in the Statement of Profit and Loss.

8. DIRECTORY, BRANCH & ADDRESS QUERIES:
   - When asked for the address, contact, or phone number of a specific branch office:
     * Locate the section for that specific city or branch in the retrieved evidence.
     * Extract the exact address, PIN code, and contact information strictly belonging to that branch.
   - When asked for branch counts, count them strictly from the retrieved directory tables.

9. BALANCE SHEET, EQUITY & NET WORTH:
   - "Total Assets" is the final aggregate row on the face of the Balance Sheet.
   - "Financial Assets" is the primary sub-grouping under Assets on the Balance Sheet.
   - "Net Worth" / "Total Equity":
     * If an explicit line item or disclosure for "Net worth" exists in the evidence, report that printed figure directly.
     * Under Equity on the Balance Sheet, Total Equity consists of Equity Share Capital + Other Equity.
     * Report the exact figures printed in the evidence.

10. YEAR GROUNDING & CORPORATE APPOINTMENTS:
    - When asked about an auditor, director, officer, salary, or event for a specific year or annual report:
      * Report the details directly from the retrieved evidence corresponding to that annual report and fiscal year.
      * When notice resolutions or appointment terms are provided (e.g., terms of appointment or re-appointment for the Managing Director or officers), report the stated salary and remuneration terms.

11. SINGLE VS MULTI-PERIOD GROUNDING:
    - Single Financial Year Notation (e.g. any hyphenated or slashed fiscal year range like "FY YYYY-YYYY" or "FY YYYY-YY") refers to a single accounting year ending in March of the later year. Provide figures only for that requested fiscal year. Do not include unrequested future or previous comparative years unless explicitly queried.
    - When multiple distinct years are queried with 'AND':
      * Provide the reported figures for each requested year separately in an itemized list or table.
      * Never perform arithmetic addition across annual periods unless explicitly requested by the user.

12. TOPIC, NOTES & ARITHMETIC CALCULATIONS:
    - Match the exact section or Note title corresponding to the queried topic directly from the retrieved document headers and tables.
    - Do not substitute numbers from unrelated schedules or adjacent chunks.
    - When asked for the totals of separate notes or line items:
      * Report the separate reported total of each individual note or line item.
      * Do NOT perform arithmetic addition, subtraction, division, or multiplication between separate line items or notes unless the user explicitly requests an arithmetic calculation using words such as "add", "sum together", "subtract", "difference", "multiply", or "divide". If the user asks for the totals of separate items, provide each item's reported total directly.

13. ZERO HALLUCINATION POLICY:
    - Never invent numbers, estimates, or assumptions.
    - Every numerical value, line item, and fact must be dynamically retrieved from the verified evidence.

14. DIVIDENDS & CORPORATE DISTRIBUTIONS:
    - When asked for "Total dividend", "dividend", or "dividends" for a fiscal year:
      * Check the dividend disclosures (such as "Details of Dividends proposed" under Note on Share Capital or "Dividend paid" under Other Equity).
      * Extract the dividend rate/percentage, dividend per share, and total dividend amount for the requested period.
      * Distinguish between proposed dividends, actual dividends paid, and unclaimed dividends.
      * Do not declare dividend information missing if the dividend schedule or dividend line item is present in the retrieved evidence.

15. DIRECTOR & EXECUTIVE REMUNERATION, SALARY & TERMS:
    - When asked specifically for "salary per month" or "monthly salary" for a Director or Managing Director:
      * Extract the exact fixed monthly salary directly stated in the appointment / re-appointment terms (e.g. "Salary ` 3.00 lakhs per month" or "` 3.00 Lakhs per month").
      * Report the monthly salary directly as the answer.
      * CRITICAL: Do NOT perform manual division of the total annual remuneration by 12 months (e.g. dividing total remuneration like 130.71 lakhs by 12 to invent a synthetic monthly figure). That is NOT the stated monthly salary because annual remuneration includes commission on net profits and perquisites.
      * If the user specifically asks for "salary per month" or "monthly salary", report ONLY the monthly salary in the table and answer as requested. Do NOT add unsolicited annual remuneration or calculations unless the user asks for total remuneration.
"""



class AnswerGenerator:
    def __init__(self):
        self.ollama = OllamaClient()

    def generate_answer(
        self,
        question: str,
        retrieved_chunks: list[DocumentChunk],
        conversation_history: Optional[list[dict[str, str]]] = None,
    ) -> AnswerResult:
        start_time = time.time()

        if not retrieved_chunks:
            return AnswerResult(
                question=question,
                answer="No relevant financial documents or source records were found to answer this question.",
                citations=[],
                confidence="unsupported",
                latency_ms=(time.time() - start_time) * 1000,
            )

        # Build context blocks
        context_blocks = []
        for idx, chunk in enumerate(retrieved_chunks, start=1):
            header = f"[Evidence {idx}] (Document: {chunk.document_filename or 'Annual Report'}, Page {chunk.page_start}"
            if chunk.section_title:
                header += f", Section/Table: {chunk.section_title}"
            if chunk.financial_period:
                header += f", Period: {chunk.financial_period}"
            if chunk.unit:
                header += f", Unit: {chunk.unit}"
            header += ")"
            context_blocks.append(f"{header}\n{chunk.normalized_text}")

        context_str = "\n\n--------------------\n\n".join(context_blocks)

        # Build prompt
        prompt = f"RETRIEVED SOURCE EVIDENCE:\n\n{context_str}\n\n"

        if conversation_history:
            # Include recent user questions so prior intent/follow-up topics guide retrieval,
            # but omit verbose assistant responses so prior financial tables/years don't bleed into new queries.
            recent_user_queries = [t['content'].strip() for t in conversation_history if t.get('role') == 'user'][-3:]
            if recent_user_queries:
                prompt += "PREVIOUS USER QUESTIONS IN SESSION:\n"
                for q in recent_user_queries:
                    prompt += f"- {q}\n"
                prompt += "\n"

        # Analyze query intent using decision engine
        from ragcli.decision_engine import DecisionEngine
        intent = DecisionEngine.analyze_query(question)

        prompt += f"USER QUESTION: {question}\n\n"
        # Detect whether the user asked for multiple distinct fiscal periods vs a single financial year
        import re
        q_lower = question.lower()
        period_spans = re.findall(r"(?:fy\s*)?20[1-2][0-9](?:[-/](?:fy\s*)?(?:20)?[0-3][0-9])?", q_lower)
        has_and_between_years = bool(re.search(r"(?:20\d{2}|fy\s*\d{2,4})\s+(?:to|and)\s+(?:20\d{2}|fy\s*\d{2,4})", q_lower))
        is_multi_year = len(period_spans) > 1 or has_and_between_years
        
        if is_multi_year:
            prompt += "INSTRUCTION FOR MULTI-YEAR QUERY:\n"
            prompt += "- The user is asking for MULTIPLE distinct financial years.\n"
            prompt += "- You are provided with multiple Evidence blocks above, each containing the financial statement from a different Annual Report:\n"
            prompt += "  * Inspect [Evidence 1], [Evidence 2], [Evidence 3], [Evidence 4], [Evidence 5], [Evidence 6] etc.\n"
            prompt += "  * Each Evidence block header indicates the Annual Report document and the financial period covered.\n"
            prompt += "  * Inside each statement, locate the line 'Profit for the year' (or the requested financial line item) and read the figures for each year.\n"
            prompt += "- You MUST report the reported figure for EVERY requested financial year in a comprehensive, professional Markdown table with columns: `Financial Year`, `Amount (` Lakhs)`, `Source Document`.\n"
            prompt += "- Do NOT declare figures missing when an Evidence block corresponding to that annual report or financial period is present in the prompt above.\n"
            prompt += "- DO NOT sum or aggregate the years into a single total unless the user specifically requested 'sum' or 'add together'.\n\n"
        elif period_spans:
            # Single financial year requested
            prompt += "INSTRUCTION FOR SINGLE FISCAL YEAR QUERY:\n"
            prompt += "- The user is asking for a SINGLE financial year.\n"
            prompt += "- Provide the exact reported figures from that requested target financial year's annual report.\n"
            prompt += "- Answer strictly what the user asked: if the user asks specifically for 'salary per month' or 'monthly salary', report ONLY the stated monthly salary (do NOT divide annual totals by 12, and do NOT add unrequested annual figures into the table).\n"
            prompt += "- If the annual report contains executive salary terms or re-appointment resolutions governing the Managing Director / Director (e.g. 'Salary ` X per month'), extract that exact monthly figure directly.\n"
            prompt += "- When asked for separate notes or line items, report each item's reported total directly in a clean table. Do NOT perform arithmetic addition between separate items unless the user explicitly used arithmetic operation words like 'add', 'plus', or 'sum together'.\n\n"



        if intent.is_calculation_requested and intent.target_metric:
            prompt += f"SYSTEM INSTRUCTION: The user is requesting calculation for {intent.target_metric.metric_name}. "
            if intent.target_metric.formula_expression:
                prompt += f"Under Ind AS NBFC rules, EBITDA is not a reported statutory line item on the face of the financial statements and MUST be derived using: {intent.target_metric.formula_expression}. "
            prompt += "Extract Profit before Tax, Finance Costs, and Depreciation & Amortization Expense for each requested period from the Statement of Profit and Loss, show the step-by-step arithmetic addition, and state the final result clearly.\n\n"

        if bool(re.search(r"(?i)\bnotes?\b", q_lower)) or "branch" in q_lower or "branches" in q_lower or "director" in q_lower:
            prompt += "PRESENTATION FORMAT INSTRUCTION:\n"
            prompt += "- Present the figures / items in a clean, professional, fully-formatted Markdown table.\n"
            prompt += "- Ensure columns have balanced headers (e.g. `| Item / Designation | Amount / Detail |` or `| Particulars | Figure |`), aligned separators (`| :--- | :---: |`), and one row per item.\n\n"

        prompt += "ANSWER (following mandatory audit rules):"

        try:
            raw_response = self.ollama.generate(
                prompt=prompt,
                system_prompt=SYSTEM_PROMPT,
                temperature=0.0,
            )
            citations = build_citations(retrieved_chunks)

            confidence = "verified"
            if "do not contain sufficient evidence" in raw_response.lower():
                confidence = "unsupported"

            from ragcli.formula_engine import DeterministicFormulaEngine
            audited_response = DeterministicFormulaEngine.verify_and_correct_arithmetic(raw_response)

            return AnswerResult(
                question=question,
                answer=audited_response,
                citations=citations,
                confidence=confidence,
                latency_ms=(time.time() - start_time) * 1000,
            )
        except Exception as e:
            logger.error(f"Error calling LLM: {e}")
            return AnswerResult(
                question=question,
                answer=f"Error generating answer from local model: {e}",
                citations=[],
                confidence="uncertain",
                latency_ms=(time.time() - start_time) * 1000,
            )
