-- Database Initialization Script for Sakthi Finance RAG

-- 1. Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Documents table
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    filename TEXT NOT NULL,
    file_hash TEXT NOT NULL UNIQUE,
    page_count INTEGER NOT NULL,
    file_size_bytes BIGINT NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Index on file_hash to prevent duplicate ingestion
CREATE INDEX IF NOT EXISTS idx_documents_file_hash ON documents(file_hash);

-- 3. Document Pages table (inspection & quality traceability)
CREATE TABLE IF NOT EXISTS document_pages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page_number INTEGER NOT NULL,
    has_text BOOLEAN NOT NULL DEFAULT TRUE,
    is_scanned BOOLEAN NOT NULL DEFAULT FALSE,
    table_count INTEGER NOT NULL DEFAULT 0,
    character_count INTEGER NOT NULL DEFAULT 0,
    extraction_method TEXT NOT NULL, -- 'native', 'layout', 'ocr'
    warnings JSONB DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_doc_page UNIQUE (document_id, page_number)
);

-- 4. Document Chunks table (financial-aware chunks)
CREATE TABLE IF NOT EXISTS document_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    chunk_type TEXT NOT NULL, -- 'narrative', 'table', 'footnote', 'header'
    page_start INTEGER NOT NULL,
    page_end INTEGER NOT NULL,
    section_title TEXT,
    table_id TEXT,
    financial_period TEXT,    -- e.g. 'FY 2024-25'
    metric_name TEXT,         -- e.g. 'Profit After Tax'
    unit TEXT,                -- e.g. '₹ in Lakhs'
    source_text TEXT NOT NULL,
    normalized_text TEXT NOT NULL,
    table_data JSONB,         -- structured representation of table if applicable
    tsv_content TSVECTOR GENERATED ALWAYS AS (
        to_tsvector('english', coalesce(normalized_text, '') || ' ' || coalesce(section_title, '') || ' ' || coalesce(financial_period, ''))
    ) STORED,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_doc_chunk_idx UNIQUE (document_id, chunk_index)
);

-- 5. Embeddings table
CREATE TABLE IF NOT EXISTS chunk_embeddings (
    chunk_id UUID PRIMARY KEY REFERENCES document_chunks(id) ON DELETE CASCADE,
    embedding_model TEXT NOT NULL,
    embedding_dim INTEGER NOT NULL,
    embedding vector(768) NOT NULL, -- configured to match model dimensions (default 768)
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 6. Indexes for hybrid retrieval
CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON document_chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_type ON document_chunks(chunk_type);
CREATE INDEX IF NOT EXISTS idx_chunks_financial_period ON document_chunks(financial_period);
CREATE INDEX IF NOT EXISTS idx_chunks_tsv ON document_chunks USING GIN(tsv_content);

-- HNSW Vector Index for Cosine Similarity (<=>)
CREATE INDEX IF NOT EXISTS idx_chunk_embeddings_hnsw 
ON chunk_embeddings USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
