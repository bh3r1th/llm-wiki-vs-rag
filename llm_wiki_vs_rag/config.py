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
    allow_rag_fallback: bool = False


class BenchmarkConfig(BaseModel):
    """Configuration for locked benchmark contract behavior."""

    locked: bool = True


class EvalConfig(BaseModel):
    """Configuration for evaluation harness and reporting."""

    metrics: list[str] = Field(default_factory=lambda: ["exact_match"])
    output_name: str = Field(default="eval_report.json", min_length=1)


class AppConfig(BaseModel):
    """Top-level application configuration."""

    project_root: Path = Field(default_factory=lambda: Path.cwd())
    rag: RAGConfig = Field(default_factory=RAGConfig)
    wiki: WikiConfig = Field(default_factory=WikiConfig)
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)

    def retrieval_top_k(self) -> int:
        """Shared retrieval budget policy for query-time benchmark comparisons."""
        if self.benchmark.locked and self.rag.top_k != self.wiki.query_top_k:
            raise ValueError(
                "Locked benchmark mode requires retrieval parity: "
                f"rag.top_k={self.rag.top_k} must equal wiki.query_top_k={self.wiki.query_top_k}."
            )
        return self.rag.top_k

    def wiki_fallback_enabled(self, requested_fallback: bool | None = None) -> bool:
        """Resolve wiki fallback behavior with strict benchmark purity guardrails."""
        enabled = self.wiki.allow_rag_fallback if requested_fallback is None else requested_fallback
        if self.benchmark.locked and enabled:
            raise ValueError("Locked benchmark mode forbids wiki->RAG fallback.")
        return enabled
