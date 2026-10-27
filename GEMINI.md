# Project Rules & Behavioral Guidelines

## 1. Zero Hardcoding Rule (STRICT)
- **Never hardcode specific questions, prompt variants, or answers.**
- **No question-specific text checks or static matching:** Never write conditionals checking for specific user questions (e.g., `if "EBITDA" in query and "2024" in query: return "..."`).
- **No hardcoded answer numbers or static text:** All numerical values, citations, line items, and report years must be dynamically retrieved, parsed, and calculated directly from the ingested documents or vector database.
- **Dynamic Parsers Only:** Any parsing logic (e.g., extracting year, line items, or tables) must rely on general regex patterns, generic semantic matching, or LLM generation over retrieved context chunks.
- **Generic Fallback / Grounding:** If information is missing from the retrieved context, the system must state that the information is unavailable or compute it dynamically using generalized mathematical / formula engines. It must never substitute pre-computed constants.

## 2. Dynamic Financial Information Extraction
- Financial formulas (like EBITDA, EBIT, Current Ratio) must parse line items dynamically from the retrieved financial statements.
- Date/Period headers (e.g., "for the year ended 31st March YYYY") must be discovered dynamically from the document chunk's text or metadata.
- Citations (document name, page number) must always reflect the actual source chunk retrieved from the database.
