"""Core typed models for benchmark data flow."""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class SourceDocument(BaseModel):
    """Raw source document loaded from disk."""

    doc_id: str = Field(min_length=1)
    source_path: Path
    text: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentBatch(BaseModel):
    """Collection of source documents for a run."""

    documents: list[SourceDocument] = Field(default_factory=list)


class QueryCase(BaseModel):
    """Single evaluation query case."""

    query_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    reference_answer: str | None = None


class RetrievedChunk(BaseModel):
    """Retrieved text unit used for synthesis."""

    doc_id: str = Field(min_length=1)
    chunk_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    source_path: Path | None = None
    position: dict[str, int] = Field(default_factory=dict)
    score: float = 0.0


class WikiPage(BaseModel):
    """Ingest-time synthesized wiki page."""

    page_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    body: str = Field(min_length=1)
    links: list[str] = Field(default_factory=list)


class GenerationResult(BaseModel):
    """Final generated answer and optional trace information."""

    query_id: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    mode: str = Field(min_length=1)
    used_context_ids: list[str] = Field(default_factory=list)
    run_id: str | None = None
    latency_ms: float | None = Field(default=None, ge=0)
    artifact_dir: str | None = None


class EvalRecord(BaseModel):
    """Evaluation record for one query."""

    query_id: str = Field(min_length=1)
    mode: str = Field(min_length=1)
    score: float = 0.0
    notes: str | None = None


class RunArtifact(BaseModel):
    """Artifact emitted by a runner step."""

    run_id: str = Field(min_length=1)
    artifact_path: Path
    artifact_type: str = Field(min_length=1)
