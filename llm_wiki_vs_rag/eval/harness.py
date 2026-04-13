"""Evaluation harness entry points."""

from __future__ import annotations

import csv
import json
from collections import defaultdict, deque
from pathlib import Path

from llm_wiki_vs_rag.config import AppConfig
from llm_wiki_vs_rag.eval.models import EvalQueryCase, EvaluationRecord, ManualEvalLabel, RunOutputRecord
from llm_wiki_vs_rag.models import QueryCase
from llm_wiki_vs_rag.paths import ProjectPaths
from llm_wiki_vs_rag.rag.pipeline import run_rag_queries
from llm_wiki_vs_rag.wiki.pipeline import run_wiki_queries


def resolve_corpus_snapshot_identity(paths: ProjectPaths, system: str) -> str:
    """Resolve a concrete corpus snapshot identifier for benchmark run outputs."""
    if system == "rag":
        manifest_path = paths.artifacts_dir / "rag_index" / "manifest.json"
    elif system == "wiki":
        manifest_path = paths.wiki_dir / "snapshot.json"
    else:
        raise ValueError(f"Unsupported system for snapshot resolution: {system}")

    if not manifest_path.exists():
        raise ValueError(f"Missing canonical snapshot manifest for {system}: {manifest_path}")

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid snapshot manifest payload for {system}: {manifest_path}")
    snapshot_id = str(payload.get("snapshot_id", "")).strip()
    if not snapshot_id:
        raise ValueError(f"Missing snapshot_id in canonical snapshot manifest for {system}: {manifest_path}")
    return snapshot_id


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
    duplicate_keys: list[tuple[str, str, str]] = []
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
            key = (label.system, label.query_id, label.phase)
            if key in labels:
                duplicate_keys.append(key)
                continue
            labels[key] = label
    if duplicate_keys:
        sample = [
            {"system": system, "query_id": query_id, "phase": phase}
            for system, query_id, phase in duplicate_keys[:5]
        ]
        raise ValueError(
            "Manual labels must be unique per (system, query_id, phase). "
            f"duplicate_sample={sample}"
        )
    return labels


def run_queries_for_system(
    config: AppConfig,
    paths: ProjectPaths,
    query_cases: list[EvalQueryCase],
    system: str,
) -> list[RunOutputRecord]:
    """Run one system over a query set and normalize outputs."""
    query_inputs = [QueryCase(query_id=case.query_id, question=case.question) for case in query_cases]
    snapshot_identity = resolve_corpus_snapshot_identity(paths=paths, system=system)

    if system == "rag":
        results = run_rag_queries(config=config, paths=paths, query_cases=query_inputs)
    elif system == "wiki":
        results = run_wiki_queries(config=config, paths=paths, query_cases=query_inputs)
    else:
        raise ValueError(f"Unsupported system: {system}")

    run_metadata = {"corpus_snapshot": snapshot_identity}
    seen_artifact_snapshots: set[str] = set()
    for result in results:
        if not result.artifact_dir:
            continue
        metadata_path = Path(result.artifact_dir) / "metadata.json"
        if not metadata_path.exists():
            continue
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid artifact metadata payload for system={system}: {metadata_path}")
        artifact_snapshot = str(payload.get("corpus_snapshot", "")).strip()
        if not artifact_snapshot:
            continue
        seen_artifact_snapshots.add(artifact_snapshot)
        if artifact_snapshot != snapshot_identity:
            raise ValueError(
                "Run snapshot attribution mismatch: "
                f"system={system}, expected={snapshot_identity}, artifact={artifact_snapshot}, path={metadata_path}."
            )
    if len(seen_artifact_snapshots) > 1:
        raise ValueError(
            "Run snapshot attribution mismatch: mixed corpus snapshots observed in per-query artifacts for "
            f"system={system}, snapshots={sorted(seen_artifact_snapshots)}."
        )

    by_query_id: dict[str, deque[EvalQueryCase]] = defaultdict(deque)
    for item in query_cases:
        by_query_id[item.query_id].append(item)

    normalized: list[RunOutputRecord] = []
    for result in results:
        queue = by_query_id.get(result.query_id)
        if not queue:
            raise ValueError(f"Run output query_id not found in query set for system={system}: {result.query_id}")
        case = queue.popleft()
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
                    "corpus_snapshot": snapshot_identity,
                    "run_corpus_snapshot": run_metadata["corpus_snapshot"],
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
