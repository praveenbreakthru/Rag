# Sakthi Finance RAG — System Architecture & File Connection Map

This document provides a comprehensive blueprint of the Sakthi Finance Financial RAG System. It details every component, file connection, and step-by-step dataflow across **Ingestion**, **Database Storage**, **Query Processing & Retrieval**, **Decision Engine / Knowledge Base Routing**, **Deterministic Calculations**, **Answer Generation**, and the **Streamlit Web UI**.

---

## 1. High-Level Architecture Diagram

```mermaid
flowchart TB
    subgraph UI_Layer["🖥️ Presentation & Interface Layer"]
        APP["UI/app.py\n(Streamlit Web Interface)"]
        CHAT_DB[("UI/chat_history.db\n(SQLite Session Memory)")]
        HIST["UI/chat_history.py\n(CRUD & Session Manager)"]
        ADAPTER["UI/rag_adapter.py\n(Singleton RAG Gateway)"]
        CLI["src/ragcli/cli.py\n(Typer Terminal CLI)"]
    end

    subgraph Config_Core["⚙️ Configuration & Schemas"]
        ENV[(".env / .env.example\n(PostgreSQL & Model Settings)")]
        CONFIG["src/ragcli/config.py\n(Pydantic Settings)"]
        SCHEMAS["src/ragcli/schemas.py\n(Type Models & DTOs)"]
    end

    subgraph Ingestion_Pipeline["📥 Document Ingestion Pipeline"]
        PDFS[("📄 6x Annual Reports (.pdf)\n(FY 2020 - FY 2025)")]
        INSP["src/ragcli/pdf_inspector.py\n(Metadata, Hash, Structure)"]
        EXTRACT["src/ragcli/pdf_extractor.py\n(PyMuPDF Page Extractor)"]
        TABLES["src/ragcli/table_extractor.py\n(Layout & Table Parser)"]
        NORM["src/ragcli/text_normalizer.py\n(Note & Unicode Sanitizer)"]
        CHUNK["src/ragcli/chunker.py\n(Financial Chunk Splitter)"]
        INGEST["src/ragcli/ingestion.py\n(Ingestion Orchestrator)"]
    end

    subgraph Storage_Layer["💾 Database & Vector Storage (PostgreSQL + pgvector)"]
        DB["src/ragcli/database.py\n(Connection & Cursor Context)"]
        REPO["src/ragcli/repository.py\n(Atomic Transactions & DDL)"]
        PG[("PostgreSQL Database: sakthi_rag\n- documents\n- document_pages\n- document_chunks\n- chunk_embeddings (768-dim)")]
    end

    subgraph Query_Engine["🔍 Retrieval & Decision Engine"]
        KB["src/ragcli/metrics_knowledge_base.py\n(120+ Financial Ontologies & Aliases)"]
        DECIDE["src/ragcli/decision_engine.py\n(Intent, Category, Period Classifier)"]
        RETR["src/ragcli/retrieval.py\n(Hybrid BM25 + Vector + Cross-Encoder)"]
        RERANK["BAAI/bge-reranker-base\n(HuggingFace Cross-Encoder)"]
    end

    subgraph AI_Generation["🧠 Reasoning & Deterministic Engine"]
        FORMULA["src/ragcli/formula_engine.py\n(Decimal Precision EBITDA / Ratios)"]
        CITE["src/ragcli/citations.py\n(Grounding & Attribution Builder)"]
        GEN["src/ragcli/answer_generator.py\n(System Prompt & Context Assembler)"]
        OLLAMA["src/ragcli/ollama_client.py\n(Ollama HTTP API Client)"]
        OLLAMA_SRV["🤖 Ollama Local Server\n- qwen2.5:14b (LLM)\n- nomic-embed-text:latest (Embedder)"]
    end

    %% Connections - UI & Ingestion Flow
    PDFS --> INSP
    INSP --> EXTRACT
    EXTRACT --> TABLES
    TABLES --> CHUNK
    EXTRACT --> NORM
    NORM --> CHUNK
    CHUNK --> INGEST
    INGEST --> OLLAMA_CLIENT_EMB["Ollama Embeddings\n(nomic-embed-text)"] --> INGEST
    INGEST --> REPO
    REPO --> DB --> PG

    %% Configuration Connections
    ENV --> CONFIG
    CONFIG --> DB
    CONFIG --> OLLAMA
    CONFIG --> RETR
    CONFIG --> GEN
    SCHEMAS --> INGEST
    SCHEMAS --> REPO
    SCHEMAS --> RETR
    SCHEMAS --> GEN

    %% UI Flow
    APP <--> HIST
    HIST <--> CHAT_DB
    APP --> ADAPTER
    CLI --> RETR
    CLI --> GEN
    ADAPTER --> RETR
    ADAPTER --> GEN

    %% Retrieval & Routing Flow
    RETR <--> DECIDE
    DECIDE <--> KB
    RETR --> PG
    RETR --> RERANK
    RETR --> GEN

    %% Generation & Output Flow
    GEN <--> FORMULA
    GEN --> CITE
    GEN --> OLLAMA --> OLLAMA_SRV
    GEN --> ADAPTER
    ADAPTER --> APP
```

---

## 2. File-by-File Connections & Responsibilities

### Tier 1: User Interface & Entrypoints

| File | Depends On (Imports) | Used By (Consumers) | Primary Role |
|---|---|---|---|
| [`UI/app.py`](file:///Users/breakthru/Desktop/Praveen/Rag/UI/app.py) | `chat_history.py`, `rag_adapter.py` | Browser Users (Port 8501) | Streamlit Web Chat UI, conversation selector, citation cards, and markdown rendering. |
| [`UI/chat_history.py`](file:///Users/breakthru/Desktop/Praveen/Rag/UI/chat_history.py) | `sqlite3` (`UI/chat_history.db`) | `UI/app.py` | Permanent SQLite storage for multi-session chat histories, titles, and timestamps. |
| [`UI/rag_adapter.py`](file:///Users/breakthru/Desktop/Praveen/Rag/UI/rag_adapter.py) | `retrieval.py`, `answer_generator.py` | `UI/app.py` | Singleton pattern bridge that coordinates retrieval and answer generation for the UI without altering backend logic. |
| [`src/ragcli/cli.py`](file:///Users/breakthru/Desktop/Praveen/Rag/src/ragcli/cli.py) | `ingestion.py`, `retrieval.py`, `answer_generator.py`, `database.py` | Terminal Users (`ragcli query`) | Typer-based Command-Line Interface providing terminal search, ingestion progress, and DB health inspection. |

---

### Tier 2: Document Ingestion Pipeline

| File | Depends On (Imports) | Used By (Consumers) | Primary Role |
|---|---|---|---|
| [`src/ragcli/pdf_inspector.py`](file:///Users/breakthru/Desktop/Praveen/Rag/src/ragcli/pdf_inspector.py) | `schemas.py`, `fitz` (PyMuPDF) | `pdf_extractor.py` | Computes SHA-256 file hashes for deduplication and extracts PDF document page counts and dimensions. |
| [`src/ragcli/table_extractor.py`](file:///Users/breakthru/Desktop/Praveen/Rag/src/ragcli/table_extractor.py) | `schemas.py`, `fitz` | `pdf_extractor.py` | Identifies tabular line bounding boxes, parses column headers, and reconstructs Markdown financial tables. |
| [`src/ragcli/text_normalizer.py`](file:///Users/breakthru/Desktop/Praveen/Rag/src/ragcli/text_normalizer.py) | Standard regex | `pdf_extractor.py`, `retrieval.py` | Normalizes OCR artefacts, cleans Note numbering (e.g. `Note 25`), currency characters (`₹`, `` ` ``), and Indian commas. |
| [`src/ragcli/pdf_extractor.py`](file:///Users/breakthru/Desktop/Praveen/Rag/src/ragcli/pdf_extractor.py) | `pdf_inspector.py`, `table_extractor.py`, `text_normalizer.py`, `schemas.py` | `ingestion.py` | Orchestrates page-by-page narrative extraction alongside reconstructed tables while preserving reading order. |
| [`src/ragcli/chunker.py`](file:///Users/breakthru/Desktop/Praveen/Rag/src/ragcli/chunker.py) | `schemas.py`, `config.py` | `ingestion.py` | Financial-aware chunker: keeps tables atomic, splits narrative paragraphs cleanly, and tags chunks with fiscal years and page numbers. |
| [`src/ragcli/ingestion.py`](file:///Users/breakthru/Desktop/Praveen/Rag/src/ragcli/ingestion.py) | `pdf_extractor.py`, `chunker.py`, `ollama_client.py`, `repository.py` | `cli.py` | Complete ingestion orchestrator: parses PDF &rarr; creates chunks &rarr; generates `nomic-embed-text` embeddings &rarr; writes to DB. |

---

### Tier 3: Storage & Database Access Layer

| File | Depends On (Imports) | Used By (Consumers) | Primary Role |
|---|---|---|---|
| [`src/ragcli/config.py`](file:///Users/breakthru/Desktop/Praveen/Rag/src/ragcli/config.py) | `pydantic_settings`, `.env` | Entire codebase | Validates environment variables (`DB_NAME`, `LLM_MODEL`, `EMBEDDING_DIM`, `RETRIEVAL_TOP_K`, `RERANKER_PATH`). |
| [`src/ragcli/database.py`](file:///Users/breakthru/Desktop/Praveen/Rag/src/ragcli/database.py) | `psycopg`, `pgvector`, `config.py` | `repository.py`, `retrieval.py` | Manages raw PostgreSQL connections, transactional cursor contexts (`get_db_cursor`), and registers vector types. |
| [`src/ragcli/repository.py`](file:///Users/breakthru/Desktop/Praveen/Rag/src/ragcli/repository.py) | `database.py`, `schemas.py` | `ingestion.py` | Manages schema creation (`documents`, `document_pages`, `document_chunks`, `chunk_embeddings`) and atomic batch inserts. |

---

### Tier 4: Knowledge Base, Retrieval & Decision Routing

| File | Depends On (Imports) | Used By (Consumers) | Primary Role |
|---|---|---|---|
| [`src/ragcli/metrics_knowledge_base.py`](file:///Users/breakthru/Desktop/Praveen/Rag/src/ragcli/metrics_knowledge_base.py) | `schemas.py` | `decision_engine.py`, `retrieval.py` | **Master Catalogue**: Contains 120+ standard Ind AS metrics, aliases, statutory categories, and appointment schemas. Zero hardcoded question/answer numbers. |
| [`src/ragcli/decision_engine.py`](file:///Users/breakthru/Desktop/Praveen/Rag/src/ragcli/decision_engine.py) | `metrics_knowledge_base.py` | `retrieval.py`, `answer_generator.py` | Query intent classifier: determines target metric, comparison periods, calculation needs, and statement type. |
| [`src/ragcli/retrieval.py`](file:///Users/breakthru/Desktop/Praveen/Rag/src/ragcli/retrieval.py) | `database.py`, `ollama_client.py`, `decision_engine.py`, `metrics_knowledge_base.py`, Cross-Encoder | `rag_adapter.py`, `cli.py` | **Hybrid Search Engine**: Combines full-text BM25 (`tsquery`), dense vector similarity (`<->`), Knowledge Base routing, and BGE cross-encoder reranking. |

---

### Tier 5: Generation, Calculations & Citations

| File | Depends On (Imports) | Used By (Consumers) | Primary Role |
|---|---|---|---|
| [`src/ragcli/formula_engine.py`](file:///Users/breakthru/Desktop/Praveen/Rag/src/ragcli/formula_engine.py) | `decimal.Decimal` | `answer_generator.py` | Precision deterministic mathematical engine for derived metrics (EBITDA, CRAR, Net Profit Margin) without LLM hallucinations. |
| [`src/ragcli/citations.py`](file:///Users/breakthru/Desktop/Praveen/Rag/src/ragcli/citations.py) | `schemas.py` | `answer_generator.py` | Assembles metadata badges (Document Name, Page Number, Section Title, Table ID, Financial Period). |
| [`src/ragcli/ollama_client.py`](file:///Users/breakthru/Desktop/Praveen/Rag/src/ragcli/ollama_client.py) | `httpx`, `config.py` | `ingestion.py`, `retrieval.py`, `answer_generator.py` | Local Ollama REST client for embedding generation (`nomic-embed-text`) and LLM streaming completion (`qwen2.5:14b`). |
| [`src/ragcli/answer_generator.py`](file:///Users/breakthru/Desktop/Praveen/Rag/src/ragcli/answer_generator.py) | `ollama_client.py`, `decision_engine.py`, `citations.py`, `formula_engine.py` | `rag_adapter.py`, `cli.py` | Constructs audit-grade system prompt, organizes multi-evidence context, enforces Markdown tables, and calls LLM. |

---

## 3. End-to-End Query Lifecycle (Step-by-Step)

```mermaid
sequenceDiagram
    autonumber
    actor User as 👤 User / Analyst
    participant UI as 🖥️ UI/app.py
    participant Adapter as 🔌 UI/rag_adapter.py
    participant Retriever as 🔍 src/ragcli/retrieval.py
    participant Decision as 🧭 src/ragcli/decision_engine.py
    participant KB as 📚 src/ragcli/metrics_knowledge_base.py
    participant PG as 💾 PostgreSQL (sakthi_rag)
    participant Rerank as ⚖️ BAAI/bge-reranker-base
    participant Generator as 🧠 src/ragcli/answer_generator.py
    participant Ollama as 🤖 Ollama (qwen2.5:14b)

    User->>UI: Types: "What was the profit for the year for fy2020-21?"
    UI->>Adapter: ask(question, conversation_history)
    
    %% Intent & Knowledge Base
    Adapter->>Retriever: retrieve(query)
    Retriever->>Decision: analyze_query(query)
    Decision->>KB: lookup_metric_by_query(query)
    KB-->>Decision: MetricDefinition(PROFIT_FOR_YEAR, aliases, P&L)
    Decision-->>Retriever: QueryIntent(metric, target_years=['2021'])

    %% Hybrid Search
    par Vector Search
        Retriever->>Ollama: get_embedding(query)
        Ollama-->>Retriever: 768-dim float vector
        Retriever->>PG: SELECT chunks ORDER BY embedding <-> query_vector LIMIT 50
        PG-->>Retriever: Top vector chunks
    and BM25 Fulltext Search
        Retriever->>PG: SELECT chunks, ts_rank(tsv_content, phraseto_tsquery)
        PG-->>Retriever: Top keyword chunks
    end

    %% Scoring & Reranking
    Retriever->>Retriever: Reciprocal Rank Fusion (RRF) + Year Filter
    Retriever->>Rerank: rerank(query, candidate_chunks)
    Rerank-->>Retriever: Sorted Top Chunks (P&L Page 61 from SFL-AR-Fullset-2021.pdf)
    Retriever-->>Adapter: Final Curated Chunks

    %% Generation
    Adapter->>Generator: generate_answer(question, chunks)
    Generator->>Generator: Assemble Context [Evidence 1..N] + System Audit Prompt
    Generator->>Ollama: generate(prompt, model='qwen2.5:14b', temp=0.0)
    Ollama-->>Generator: Raw Synthesized Response
    Generator->>Generator: Attach Citations (citations.py)
    Generator-->>Adapter: AnswerResult(answer, citations, confidence, latency)
    Adapter-->>UI: Output Payload
    UI->>User: Displays Answer + Citation Badges + Metrics
```
