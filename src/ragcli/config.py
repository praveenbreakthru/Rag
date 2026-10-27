"""
Configuration management for Sakthi Finance RAG system.
Loads settings from environment variables and .env file.
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    # Database
    db_host: str = Field(default="localhost", alias="DB_HOST")
    db_port: int = Field(default=5432, alias="DB_PORT")
    db_name: str = Field(default="sakthi_rag", alias="DB_NAME")
    db_user: str = Field(default="breakthru", alias="DB_USER")
    db_password: str = Field(default="", alias="DB_PASSWORD")

    # Ollama LLM & Embeddings
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    llm_model: str = Field(default="qwen2.5:14b", alias="LLM_MODEL")
    embedding_model: str = Field(default="nomic-embed-text:latest", alias="EMBEDDING_MODEL")
    embedding_dim: int = Field(default=768, alias="EMBEDDING_DIM")

    # Chunking
    chunk_size: int = Field(default=1200, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, alias="CHUNK_OVERLAP")
    table_max_rows_per_chunk: int = Field(default=30, alias="TABLE_MAX_ROWS_PER_CHUNK")

    # Retrieval & Reranking
    retrieval_top_k: int = Field(default=5, alias="RETRIEVAL_TOP_K")
    hybrid_alpha: float = Field(default=0.5, alias="HYBRID_ALPHA")
    use_reranker: bool = Field(default=True, alias="USE_RERANKER")
    reranker_model: str = Field(default="BAAI/bge-reranker-base", alias="RERANKER_MODEL")
    reranker_path: str = Field(
        default="/Users/breakthru/.cache/huggingface/hub/models--BAAI--bge-reranker-base/snapshots/2cfc18c9415c912f9d8155881c133215df768a70",
        alias="RERANKER_PATH"
    )

    # Paths & Logging
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_file: str = Field(default="logs/rag.log", alias="LOG_FILE")
    data_dir: Path = Field(default=Path("data"), alias="DATA_DIR")
    documents_dir: Path = Field(default=Path("documents"), alias="DOCUMENTS_DIR")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def database_url(self) -> str:
        """Construct PostgreSQL connection string."""
        auth = self.db_user
        if self.db_password:
            auth += f":{self.db_password}"
        return f"postgresql://{auth}@{self.db_host}:{self.db_port}/{self.db_name}"


settings = Settings()
