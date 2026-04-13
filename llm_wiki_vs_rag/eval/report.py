"""Evaluation report writers."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from llm_wiki_vs_rag.eval.metrics import compute_drift, summarize_records
from llm_wiki_vs_rag.eval.models import ComparisonReport, EvaluationRecord


def build_comparison_report(records: list[EvaluationRecord]) -> ComparisonReport:
    """Build report model with grouped summaries and drift deltas."""
    return ComparisonReport(
        summaries_by_system=summarize_records(records, group_fields=("system",)),
        summaries_by_phase=summarize_records(records, group_fields=("phase",)),
        summaries_by_category=summarize_records(records, group_fields=("category",)),
        drifts=compute_drift(records),
    )


def write_reports(records: list[EvaluationRecord], output_dir: Path) -> None:
    """Emit summary JSON/CSV, per-query CSV and markdown report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison = build_comparison_report(records)

    (output_dir / "summary.json").write_text(json.dumps(comparison.model_dump(mode="json"), indent=2), encoding="utf-8")

    _write_summary_csv(comparison=comparison, output_path=output_dir / "summary.csv")
    _write_per_query_csv(records=records, output_path=output_dir / "per_query_results.csv")
    (output_dir / "report.md").write_text(_render_markdown_report(comparison=comparison), encoding="utf-8")


def _write_summary_csv(comparison: ComparisonReport, output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dimension",
                "key",
                "total",
                "labeled_total",
                "accuracy_correct_pct",
                "latest_state_correct_pct",
                "contradiction_resolved_pct",
                "avg_latency_ms",
                "avg_total_tokens",
            ],
        )
        writer.writeheader()
        for dimension, summaries in [
            ("system", comparison.summaries_by_system),
            ("phase", comparison.summaries_by_phase),
            ("category", comparison.summaries_by_category),
        ]:
            for summary in summaries:
                writer.writerow(
                    {
                        "dimension": dimension,
                        "key": next(iter(summary.group_by.values())),
                        "total": summary.total,
                        "labeled_total": summary.labeled_total,
                        "accuracy_correct_pct": summary.metrics.get("accuracy", {}).get("correct_pct", ""),
                        "latest_state_correct_pct": summary.metrics.get("latest_state", {}).get("correct_pct", ""),
                        "contradiction_resolved_pct": summary.metrics.get("contradiction", {}).get("resolved_pct", ""),
                        "avg_latency_ms": "" if summary.avg_latency_ms is None else summary.avg_latency_ms,
                        "avg_total_tokens": "" if summary.avg_total_tokens is None else summary.avg_total_tokens,
                    }
                )


def _write_per_query_csv(records: list[EvaluationRecord], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "query_id",
                "system",
                "run_id",
                "phase",
                "category",
                "question",
                "answer",
                "artifact_dir",
                "latency_ms",
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "accuracy",
                "synthesis",
                "latest_state",
                "contradiction_detected",
                "contradiction_resolved",
                "compression_loss",
                "provenance_fidelity",
                "evaluator_notes",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "query_id": record.query_id,
                    "system": record.system,
                    "run_id": record.run_id or "",
                    "phase": record.phase,
                    "category": record.category,
                    "question": record.question,
                    "answer": record.answer,
                    "artifact_dir": str(record.metadata.get("artifact_dir", "")),
                    "latency_ms": "" if record.latency_ms is None else record.latency_ms,
                    "prompt_tokens": "" if record.prompt_tokens is None else record.prompt_tokens,
                    "completion_tokens": "" if record.completion_tokens is None else record.completion_tokens,
                    "total_tokens": "" if record.total_tokens is None else record.total_tokens,
                    "accuracy": record.accuracy or "",
                    "synthesis": record.synthesis or "",
                    "latest_state": record.latest_state or "",
                    "contradiction_detected": record.contradiction_detected,
                    "contradiction_resolved": record.contradiction_resolved,
                    "compression_loss": record.compression_loss or "",
                    "provenance_fidelity": record.provenance_fidelity,
                    "evaluator_notes": record.evaluator_notes,
                }
            )


def _render_markdown_report(comparison: ComparisonReport) -> str:
    lines = [
        "# RAG vs Wiki Evaluation Report",
        "",
        "## System Summary",
        "",
        "| System | N | Labeled | Accuracy % | Latest-State % | Contradiction Resolved % | Avg Latency (ms) | Avg Tokens |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in comparison.summaries_by_system:
        lines.append(
            "| {system} | {total} | {labeled} | {acc} | {latest} | {cr} | {lat} | {tok} |".format(
                system=summary.group_by["system"],
                total=summary.total,
                labeled=summary.labeled_total,
                acc=summary.metrics.get("accuracy", {}).get("correct_pct", "-"),
                latest=summary.metrics.get("latest_state", {}).get("correct_pct", "-"),
                cr=summary.metrics.get("contradiction", {}).get("resolved_pct", "-"),
                lat=summary.avg_latency_ms if summary.avg_latency_ms is not None else "-",
                tok=summary.avg_total_tokens if summary.avg_total_tokens is not None else "-",
            )
        )

    lines.extend(
        [
            "",
            "## Drift (Phase 2 - Phase 1)",
            "",
            "| System | Category | Accuracy Δ | Latest-State Δ | Contradiction-Resolved Δ |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for drift in comparison.drifts:
        lines.append(
            f"| {drift.system} | {drift.category} | {drift.accuracy_correct_rate_delta if drift.accuracy_correct_rate_delta is not None else '-'} | {drift.latest_state_correct_rate_delta if drift.latest_state_correct_rate_delta is not None else '-'} | {drift.contradiction_resolved_rate_delta if drift.contradiction_resolved_rate_delta is not None else '-'} |"
        )

    lines.append("")
    return "\n".join(lines)
