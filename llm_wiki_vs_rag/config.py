"""Configuration models for application and pipeline modes."""

from pathlib import Path

from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    """Configuration for the LLM client abstraction."""

    provider: str = Field(default="stub", min_length=1)
    model_name: str = Field(default="dummy-model", min_length=1)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    timeout_seconds: int = Field(default=30, ge=1)


class RAGConfig(BaseModel):
    """Configuration for query-time synthesis (RAG)."""

    chunk_size: int = Field(default=500, ge=1)
    chunk_overlap: int = Field(default=50, ge=0)
    top_k: int = Field(default=5, ge=1)


class WikiConfig(BaseModel):
    """Configuration for ingest-time synthesis (LLM Wiki)."""

    page_token_budget: int = Field(default=1200, ge=1)
    max_links_per_page: int = Field(default=5, ge=0)


class EvalConfig(BaseModel):
    """Configuration for evaluation harness and reporting."""

    metrics: list[str] = Field(default_factory=lambda: ["exact_match"])
    output_name: str = Field(default="eval_report.json", min_length=1)


class AppConfig(BaseModel):
    """Top-level application configuration."""

    project_root: Path = Field(default_factory=lambda: Path.cwd())
    rag: RAGConfig = Field(default_factory=RAGConfig)
    wiki: WikiConfig = Field(default_factory=WikiConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
