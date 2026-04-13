"""Evaluation-specific typed models."""

from pydantic import BaseModel, Field


class EvalSummary(BaseModel):
    """Aggregate evaluation summary."""

    total: int = Field(default=0, ge=0)
    mean_score: float = 0.0
