"""Evaluation harness entry points."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from llm_wiki_vs_rag.config import AppConfig
from llm_wiki_vs_rag.eval.models import EvalQueryCase, EvaluationRecord, ManualEvalLabel, RunOutputRecord
from llm_wiki_vs_rag.models import QueryCase
from llm_wiki_vs_rag.paths import ProjectPaths
from llm_wiki_vs_rag.rag.pipeline import run_rag_queries
from llm_wiki_vs_rag.wiki.pipeline import run_wiki_queries


def load_query_cases(path: Path) -> list[EvalQueryCase]:
    """Load evaluation query cases from JSON or JSONL."""
    if path.suffix == ".jsonl":
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return [EvalQueryCase.model_validate(json.loads(line)) for line in lines]

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("queries", [])
    return [EvalQueryCase.model_validate(item) for item in payload]


def _to_bool(raw: str) -> bool:
    return raw.strip().lower() in {"1", "true", "t", "yes", "y"}


def load_manual_labels(csv_path: Path) -> dict[tuple[str, str, str], ManualEvalLabel]:
    """Load manual human labels from CSV keyed by (system, query_id, phase)."""
    labels: dict[tuple[str, str, str], ManualEvalLabel] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            system = (row.get("system") or "").strip()
            if not system or system == "*":
                raise ValueError(
                    "Manual labels must specify an explicit system per row; wildcard labels are not supported."
                )
            phase = (row.get("phase") or "").strip()
            if not phase:
                raise ValueError("Manual labels must include a phase value for phase-based evaluation.")
            label = ManualEvalLabel(
                query_id=row["query_id"],
                system=(system or None),
                phase=phase,
                accuracy=row["accuracy"],
                synthesis=row["synthesis"],
                latest_state=row["latest_state"],
                contradiction_detected=_to_bool(row["contradiction_detected"]),
                contradiction_resolved=_to_bool(row["contradiction_resolved"]),
                compression_loss=row["compression_loss"],
                provenance_fidelity=_to_bool(row["provenance_fidelity"]),
                evaluator_notes=row.get("evaluator_notes", ""),
            )
            labels[(label.system, label.query_id, label.phase)] = label
    return labels


def run_queries_for_system(
    config: AppConfig,
    paths: ProjectPaths,
    query_cases: list[EvalQueryCase],
    system: str,
) -> list[RunOutputRecord]:
    """Run one system over a query set and normalize outputs."""
    query_inputs = [QueryCase(query_id=case.query_id, question=case.question) for case in query_cases]

    if system == "rag":
        results = run_rag_queries(config=config, paths=paths, query_cases=query_inputs)
    elif system == "wiki":
        results = run_wiki_queries(config=config, paths=paths, query_cases=query_inputs)
    else:
        raise ValueError(f"Unsupported system: {system}")

    by_query = {item.query_id: item for item in query_cases}

    normalized: list[RunOutputRecord] = []
    for result in results:
        case = by_query[result.query_id]
        normalized.append(
            RunOutputRecord(
                query_id=result.query_id,
                system=system,
                phase=case.phase,
                question=case.question,
                category=case.category,
                answer=result.answer,
                run_id=result.run_id,
                latency_ms=result.latency_ms,
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                total_tokens=result.total_tokens,
                metadata={
                    "used_context_ids": result.used_context_ids,
                    "artifact_dir": result.artifact_dir,
                },
            )
        )
    return normalized


def save_run_outputs(records: list[RunOutputRecord], output_path: Path) -> None:
    """Save normalized run outputs as JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.model_dump(mode="json")) + "\n")


def load_run_outputs(path: Path) -> list[RunOutputRecord]:
    """Load normalized run outputs from JSONL or JSON."""
    if path.suffix == ".jsonl":
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return [RunOutputRecord.model_validate(json.loads(line)) for line in lines]

    payload = json.loads(path.read_text(encoding="utf-8"))
    return [RunOutputRecord.model_validate(item) for item in payload]


def merge_outputs_with_labels(
    run_outputs: list[RunOutputRecord],
    labels_by_system_query_phase: dict[tuple[str, str, str], ManualEvalLabel],
) -> list[EvaluationRecord]:
    """Merge system outputs with manual labels into evaluation records."""
    missing_keys: list[tuple[str, str, str]] = []
    for output in run_outputs:
        key = (output.system, output.query_id, output.phase)
        label = labels_by_system_query_phase.get(key)
        if label is None:
            missing_keys.append(key)
            continue
        if (
            label.accuracy is None
            or label.synthesis is None
            or label.latest_state is None
            or label.contradiction_detected is None
            or label.contradiction_resolved is None
            or label.compression_loss is None
            or label.provenance_fidelity is None
        ):
            missing_keys.append(key)
    if missing_keys:
        sample = [
            {"system": system, "query_id": query_id, "phase": phase}
            for system, query_id, phase in missing_keys[:5]
        ]
        raise ValueError(
            "Manual labels are required for every output row and must be complete. "
            f"missing_or_partial_label_sample={sample}"
        )

    records: list[EvaluationRecord] = []
    for output in run_outputs:
        label = labels_by_system_query_phase.get((output.system, output.query_id, output.phase))
        records.append(
            EvaluationRecord(
                query_id=output.query_id,
                system=output.system,
                phase=output.phase,
                question=output.question,
                category=output.category,
                answer=output.answer,
                run_id=output.run_id,
                latency_ms=output.latency_ms,
                prompt_tokens=output.prompt_tokens,
                completion_tokens=output.completion_tokens,
                total_tokens=output.total_tokens,
                metadata=output.metadata,
                accuracy=label.accuracy,
                synthesis=label.synthesis,
                latest_state=label.latest_state,
                contradiction_detected=label.contradiction_detected,
                contradiction_resolved=label.contradiction_resolved,
                compression_loss=label.compression_loss,
                provenance_fidelity=label.provenance_fidelity,
                evaluator_notes=label.evaluator_notes,
            )
        )
    return records
