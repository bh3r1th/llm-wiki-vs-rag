"""Tests for evaluation harness, metrics, drift and reporting."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from llm_wiki_vs_rag.eval.harness import (
    load_manual_labels,
    merge_outputs_with_labels,
    run_queries_for_system,
)
from llm_wiki_vs_rag.eval.metrics import compute_drift, summarize_records
from llm_wiki_vs_rag.eval.models import RunOutputRecord
from llm_wiki_vs_rag.models import GenerationResult
from llm_wiki_vs_rag.paths import ProjectPaths
from llm_wiki_vs_rag.eval.report import write_reports
from llm_wiki_vs_rag.config import AppConfig


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
                "system,query_id,accuracy,synthesis,latest_state,contradiction_detected,contradiction_resolved,compression_loss,provenance_fidelity,evaluator_notes",
                "rag,q1,correct,full,correct,true,true,none,true,looks good",
                "wiki,q2,wrong,failed,stale,false,false,major,false,bad output",
            ]
        ),
        encoding="utf-8",
    )

    labels = load_manual_labels(labels_path)

    assert labels[("rag", "q1")].accuracy == "correct"
    assert labels[("rag", "q1")].contradiction_detected is True
    assert labels[("wiki", "q2")].compression_loss == "major"


def test_metric_aggregation_and_drift():
    labels_csv = "\n".join([
        "system,query_id,accuracy,synthesis,latest_state,contradiction_detected,contradiction_resolved,compression_loss,provenance_fidelity,evaluator_notes",
        "rag,q1,correct,full,correct,true,true,none,true,",
        "rag,q2,wrong,failed,stale,false,false,major,false,",
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
                "system",
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
        writer.writerow(["q1", "rag", "correct", "full", "correct", "true", "true", "none", "true", ""])
        writer.writerow(["q2", "rag", "wrong", "failed", "stale", "false", "false", "major", "false", ""])

    records = merge_outputs_with_labels(_sample_outputs(), load_manual_labels(labels_path))
    output_dir = tmp_path / "report"
    write_reports(records, output_dir)

    assert (output_dir / "summary.json").exists()
    assert (output_dir / "summary.csv").exists()
    assert (output_dir / "per_query_results.csv").exists()
    assert (output_dir / "report.md").exists()

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert "summaries_by_system" in summary


def test_merge_uses_system_plus_query_id_for_labels():
    outputs = [
        RunOutputRecord(query_id="q1", system="rag", phase="phase_1", question="Q", category="policy", answer="A"),
        RunOutputRecord(query_id="q1", system="wiki", phase="phase_1", question="Q", category="policy", answer="A"),
    ]
    labels = {
        ("rag", "q1"): load_manual_labels_from_row("rag", "q1", "correct"),
        ("wiki", "q1"): load_manual_labels_from_row("wiki", "q1", "wrong"),
    }

    records = merge_outputs_with_labels(outputs, labels)
    by_system = {record.system: record for record in records}
    assert by_system["rag"].accuracy == "correct"
    assert by_system["wiki"].accuracy == "wrong"


def load_manual_labels_from_row(system: str, query_id: str, accuracy: str):
    return load_manual_labels_from_text(
        "\n".join(
            [
                "system,query_id,accuracy,synthesis,latest_state,contradiction_detected,contradiction_resolved,compression_loss,provenance_fidelity,evaluator_notes",
                f"{system},{query_id},{accuracy},full,correct,true,true,none,true,",
            ]
        )
    )[(system, query_id)]


def load_manual_labels_from_text(csv_payload: str):
    from tempfile import NamedTemporaryFile

    with NamedTemporaryFile("w+", suffix=".csv") as tmp:
        tmp.write(csv_payload)
        tmp.flush()
        return load_manual_labels(Path(tmp.name))


def test_run_queries_for_system_preserves_per_query_latency_and_run_id(monkeypatch, tmp_path):
    def _fake_rag(**_kwargs):
        return [
            GenerationResult(query_id="q1", answer="a1", mode="rag", run_id="r1", latency_ms=11.1, artifact_dir="art/1"),
            GenerationResult(query_id="q2", answer="a2", mode="rag", run_id="r2", latency_ms=22.2, artifact_dir="art/2"),
        ]

    monkeypatch.setattr("llm_wiki_vs_rag.eval.harness.run_rag_queries", _fake_rag)
    records = run_queries_for_system(
        config=AppConfig(project_root=tmp_path),
        paths=ProjectPaths(project_root=tmp_path),
        query_cases=[
            load_query_case("q1", "Q1"),
            load_query_case("q2", "Q2"),
        ],
        system="rag",
    )

    assert [record.latency_ms for record in records] == [11.1, 22.2]
    assert [record.run_id for record in records] == ["r1", "r2"]


def load_query_case(query_id: str, question: str):
    from llm_wiki_vs_rag.eval.models import EvalQueryCase

    return EvalQueryCase(query_id=query_id, question=question, category="policy", phase="phase_1")


def test_report_generation_contains_run_traceability_fields(tmp_path):
    records = [
        RunOutputRecord(
            query_id="q1",
            system="rag",
            phase="phase_1",
            question="Q1",
            category="policy",
            answer="A1",
            run_id="rag-1",
            latency_ms=10.0,
            metadata={"artifact_dir": "artifacts/rag_runs/rag-1"},
        )
    ]
    labels = {("rag", "q1"): load_manual_labels_from_row("rag", "q1", "correct")}
    output_dir = tmp_path / "report_traceability"
    write_reports(merge_outputs_with_labels(records, labels), output_dir)

    per_query = (output_dir / "per_query_results.csv").read_text(encoding="utf-8")
    assert "run_id" in per_query
    assert "artifact_dir" in per_query
    assert "rag-1" in per_query
