"""Evaluation report writers."""

import json
from pathlib import Path

from llm_wiki_vs_rag.models import EvalRecord


def write_eval_report(records: list[EvalRecord], output_path: Path) -> None:
    """Persist evaluation records as JSON array."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [record.model_dump() for record in records]
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
