"""Configuration models for application and pipeline modes."""

from pathlib import Path

from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    """Configuration for the LLM client abstraction."""

    provider: str = Field(default="openai-compatible", min_length=1)
    model_name: str = Field(default="dummy-model", min_length=1)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    timeout_seconds: int = Field(default=30, ge=1)
    base_url: str | None = None
    api_key: str | None = None
    mock_mode: bool = False
    mock_response: str = '{"pages_to_create": [], "pages_to_update": [], "index_note": "", "log_note": ""}'


class RAGConfig(BaseModel):
    """Configuration for query-time synthesis (RAG)."""

    chunk_size: int = Field(default=500, ge=1)
    chunk_overlap: int = Field(default=50, ge=0)
    top_k: int = Field(default=5, ge=1)


class WikiConfig(BaseModel):
    """Configuration for ingest-time synthesis (LLM Wiki)."""

    page_token_budget: int = Field(default=1200, ge=1)
    max_links_per_page: int = Field(default=5, ge=0)
    query_top_k: int = Field(default=5, ge=1)


class BenchmarkConfig(BaseModel):
    """Configuration for locked benchmark contract behavior."""

    locked: bool = True


class AppConfig(BaseModel):
    """Top-level application configuration."""

    project_root: Path = Field(default_factory=lambda: Path.cwd())
    rag: RAGConfig = Field(default_factory=RAGConfig)
    wiki: WikiConfig = Field(default_factory=WikiConfig)
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)

    def retrieval_top_k(self) -> int:
        """Shared retrieval budget policy for query-time benchmark comparisons."""
        if self.benchmark.locked and self.rag.top_k != self.wiki.query_top_k:
            raise ValueError(
                "Locked benchmark mode requires retrieval parity: "
                f"rag.top_k={self.rag.top_k} must equal wiki.query_top_k={self.wiki.query_top_k}."
            )
        return self.rag.top_k
