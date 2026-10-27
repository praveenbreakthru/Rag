# Sakthi Finance Financial Document RAG System

A production-minded, terminal-only Retrieval-Augmented Generation (RAG) system tailored specifically for Sakthi Finance financial PDFs. It preserves financial numbers, tables, row/column relationships, financial years, units (e.g. ₹ in Lakhs/Crores), and exact citations using local Ollama (`qwen2.5:7b`) and PostgreSQL with `pgvector`.

---

## Key Features

1. **Terminal-First Workflow**: Complete CLI with rich formatting via `ragcli`. No browser, React, or cloud API required.
2. **Financial Table Fidelity**: Extracts and reconstructs tables, repeating headers and column contexts across split chunks so numbers are never orphaned from their metrics.
3. **Number & Unicode Preservation**: Preserves parentheses as negative values `(250.50)`, comma groupings `1,25,00,000`, `₹` symbols, em/en-dashes, and distinguishes nil/placeholder dashes from zeros.
4. **Hybrid Retrieval**: Combines pgvector semantic cosine similarity search (`<=>`) with PostgreSQL full-text search (`tsvector` & `ts_rank`) using Reciprocal Rank Fusion (RRF).
5. **Local Qwen 2.5 7B Grounding**: Directly queries the local Ollama instance with zero temperature and a strict audit prompt that refuses to hallucinate missing financial figures.
6. **Auditable Citations**: Every answer provides the document name, PDF page number, table/section, period, and unit.

---

## Directory Structure

```text
sakthi-finance-rag/
├── pyproject.toml
├── requirements.txt
├── .env.example
├── .env
├── scripts/
│   └── init_db.sql
├── src/
│   └── ragcli/
│       ├── __init__.py
│       ├── __main__.py
│       ├── cli.py               # Typer & Rich CLI
│       ├── config.py            # Environment configuration
│       ├── database.py          # PostgreSQL + pgvector connection
│       ├── schemas.py           # Pydantic schemas
│       ├── pdf_inspector.py     # PDF inspection & scan detection
│       ├── pdf_extractor.py     # Page text & table extraction
│       ├── table_extractor.py   # Table structure reconstruction
│       ├── text_normalizer.py   # Financial numbers & Unicode normalizer
│       ├── chunker.py           # Financial-aware chunker
│       ├── ollama_client.py     # Ollama client for Qwen & embeddings
│       ├── repository.py        # Database operations & transactions
│       ├── ingestion.py         # End-to-end ingestion pipeline
│       ├── retrieval.py         # Hybrid pgvector + FTS RRF retriever
│       ├── answer_generator.py  # Grounded Qwen answer generator
│       ├── citations.py         # Citation builder
│       └── logging_config.py    # Logging
└── tests/
    ├── test_financial_numbers.py
    ├── test_chunking.py
    └── test_end_to_end.py
```

---

## Quickstart & CLI Commands

Activate the virtual environment:

```bash
cd /Users/breakthru/Desktop/Praveen/Rag
source .venv/bin/activate
```

### 1. Check System Health
Inspect Python, PostgreSQL, pgvector, and Ollama:
```bash
ragcli doctor
```

### 2. Initialize Database
Initialize the `sakthi_rag` database schema and pgvector indexes:
```bash
ragcli db init
```

### 3. Inspect a PDF Before Ingestion
Inspect page structure, extractable character counts, and tables:
```bash
ragcli inspect "./documents/annual_report.pdf"

# Preview extraction of a specific page:
ragcli inspect "./documents/annual_report.pdf" --page 12
```

### 4. Ingest Documents
Ingest a single PDF or an entire directory:
```bash
ragcli ingest "./documents/annual_report.pdf"
ragcli ingest-dir "./documents"
```

### 5. Inspect Retrieved Chunks (Without LLM)
```bash
ragcli retrieve "Profit after tax FY 2024-25" --top-k 5
```

### 6. Ask One-Time Financial Questions
```bash
ragcli ask "What was the profit after tax in FY 2024-25?"
```

### 7. Interactive Terminal Chat (With Context)
```bash
ragcli chat
```

### 8. View Statistics and Manage Documents
```bash
ragcli stats
ragcli documents list
ragcli documents delete "<document-id>"
```

---

## Running Automated Tests

Run the complete test suite:
```bash
source .venv/bin/activate
pytest
```

cd /Users/breakthru/Desktop/Praveen/Rag
source .venv/bin/activate
ragcli chat


cd UI
streamlit run app.py



cd UI
source ../.venv/bin/activate
streamlit run app.py


http://172.16.15.21:8501
