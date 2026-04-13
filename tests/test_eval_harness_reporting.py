"""Tests for evaluation harness, metrics, drift and reporting."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from llm_wiki_vs_rag.eval.harness import load_manual_labels, merge_outputs_with_labels
from llm_wiki_vs_rag.eval.metrics import compute_drift, summarize_records
from llm_wiki_vs_rag.eval.models import RunOutputRecord
from llm_wiki_vs_rag.eval.report import write_reports


def _sample_outputs() -> list[RunOutputRecord]:
    return [
        RunOutputRecord(
            query_id="q1",
            system="rag",
            phase="phase_1",
            question="Q1",
            category="policy",
            answer="A1",
            latency_ms=100.0,
            total_tokens=20,
        ),
        RunOutputRecord(
            query_id="q2",
            system="rag",
            phase="phase_2",
            question="Q2",
            category="policy",
            answer="A2",
            latency_ms=200.0,
            total_tokens=30,
        ),
    ]


def test_load_manual_labels(tmp_path):
    labels_path = tmp_path / "labels.csv"
    labels_path.write_text(
        "\n".join(
            [
                "query_id,accuracy,synthesis,latest_state,contradiction_detected,contradiction_resolved,compression_loss,provenance_fidelity,evaluator_notes",
                "q1,correct,full,correct,true,true,none,true,looks good",
                "q2,wrong,failed,stale,false,false,major,false,bad output",
            ]
        ),
        encoding="utf-8",
    )

    labels = load_manual_labels(labels_path)

    assert labels["q1"].accuracy == "correct"
    assert labels["q1"].contradiction_detected is True
    assert labels["q2"].compression_loss == "major"


def test_metric_aggregation_and_drift():
    labels_csv = "\n".join([
        "query_id,accuracy,synthesis,latest_state,contradiction_detected,contradiction_resolved,compression_loss,provenance_fidelity,evaluator_notes",
        "q1,correct,full,correct,true,true,none,true,",
        "q2,wrong,failed,stale,false,false,major,false,",
    ])
    from tempfile import NamedTemporaryFile
    with NamedTemporaryFile("w+", suffix=".csv") as tmp:
        tmp.write(labels_csv)
        tmp.flush()
        records = merge_outputs_with_labels(_sample_outputs(), load_manual_labels(Path(tmp.name)))

    by_system = summarize_records(records, group_fields=("system",))
    assert by_system[0].group_by["system"] == "rag"
    assert by_system[0].metrics["accuracy"]["correct"] == 1
    assert by_system[0].avg_latency_ms == 150.0

    drifts = compute_drift(records)
    assert len(drifts) == 1
    assert drifts[0].accuracy_correct_rate_delta == -1.0


def test_report_file_generation(tmp_path):
    labels_path = tmp_path / "labels.csv"
    with labels_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "query_id",
                "accuracy",
                "synthesis",
                "latest_state",
                "contradiction_detected",
                "contradiction_resolved",
                "compression_loss",
                "provenance_fidelity",
                "evaluator_notes",
            ]
        )
        writer.writerow(["q1", "correct", "full", "correct", "true", "true", "none", "true", ""])
        writer.writerow(["q2", "wrong", "failed", "stale", "false", "false", "major", "false", ""])

    records = merge_outputs_with_labels(_sample_outputs(), load_manual_labels(labels_path))
    output_dir = tmp_path / "report"
    write_reports(records, output_dir)

    assert (output_dir / "summary.json").exists()
    assert (output_dir / "summary.csv").exists()
    assert (output_dir / "per_query_results.csv").exists()
    assert (output_dir / "report.md").exists()

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert "summaries_by_system" in summary
