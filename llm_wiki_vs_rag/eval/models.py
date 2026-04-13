"""Evaluation-specific typed models."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

AccuracyLabel = Literal["correct", "partial", "wrong"]
SynthesisLabel = Literal["full", "incomplete", "failed"]
LatestStateLabel = Literal["correct", "stale", "missed_update"]
CompressionLossLabel = Literal["none", "minor", "major"]
PhaseLabel = Literal["phase_1", "phase_2"]
SystemLabel = Literal["rag", "wiki"]


class EvalQueryCase(BaseModel):
    """Single query case for evaluation experiments."""

    query_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    category: str = Field(min_length=1)
    phase: PhaseLabel


class RunOutputRecord(BaseModel):
    """Normalized output from one system answering one query."""

    query_id: str = Field(min_length=1)
    system: SystemLabel
    phase: PhaseLabel
    question: str = Field(min_length=1)
    category: str = Field(min_length=1)
    answer: str = ""
    run_id: str | None = None
    latency_ms: float | None = Field(default=None, ge=0)
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ManualEvalLabel(BaseModel):
    """Human-entered labels for one query response."""

    query_id: str = Field(min_length=1)
    system: SystemLabel | None = None
    phase: PhaseLabel
    accuracy: AccuracyLabel
    synthesis: SynthesisLabel
    latest_state: LatestStateLabel
    contradiction_detected: bool
    contradiction_resolved: bool
    compression_loss: CompressionLossLabel
    provenance_fidelity: bool
    evaluator_notes: str = ""


class EvaluationRecord(BaseModel):
    """Merged run output and human labels used for metric aggregation."""

    query_id: str = Field(min_length=1)
    system: SystemLabel
    phase: PhaseLabel
    question: str = Field(min_length=1)
    category: str = Field(min_length=1)
    answer: str = ""
    run_id: str | None = None
    latency_ms: float | None = Field(default=None, ge=0)
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    accuracy: AccuracyLabel | None = None
    synthesis: SynthesisLabel | None = None
    latest_state: LatestStateLabel | None = None
    contradiction_detected: bool | None = None
    contradiction_resolved: bool | None = None
    compression_loss: CompressionLossLabel | None = None
    provenance_fidelity: bool | None = None
    evaluator_notes: str = ""

    @property
    def is_labeled(self) -> bool:
        return self.accuracy is not None


class EvalSummary(BaseModel):
    """Aggregate evaluation summary for a grouping dimension."""

    group_by: dict[str, str]
    total: int = Field(default=0, ge=0)
    labeled_total: int = Field(default=0, ge=0)
    avg_latency_ms: float | None = None
    avg_total_tokens: float | None = None
    metrics: dict[str, dict[str, float | int | str | None]] = Field(default_factory=dict)


class DriftSummary(BaseModel):
    """Phase 1 vs Phase 2 deltas for key indicators."""

    system: SystemLabel
    category: str
    phase_1_count: int = Field(default=0, ge=0)
    phase_2_count: int = Field(default=0, ge=0)
    accuracy_correct_rate_delta: float | None = None
    latest_state_correct_rate_delta: float | None = None
    contradiction_resolved_rate_delta: float | None = None


class ComparisonReport(BaseModel):
    """Top-level report payload used by JSON and markdown writers."""

    summaries_by_system: list[EvalSummary] = Field(default_factory=list)
    summaries_by_phase: list[EvalSummary] = Field(default_factory=list)
    summaries_by_category: list[EvalSummary] = Field(default_factory=list)
    drifts: list[DriftSummary] = Field(default_factory=list)
