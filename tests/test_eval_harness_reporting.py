"""Tests for evaluation harness, metrics, drift and reporting."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from llm_wiki_vs_rag.eval.harness import (
    load_manual_labels,
    merge_outputs_with_labels,
    resolve_corpus_snapshot_identity,
    run_queries_for_system,
)
from llm_wiki_vs_rag.eval.metrics import compute_drift, summarize_records
from llm_wiki_vs_rag.eval.models import RunOutputRecord
from llm_wiki_vs_rag.models import GenerationResult
from llm_wiki_vs_rag.paths import ProjectPaths
from llm_wiki_vs_rag.eval.report import write_reports
from llm_wiki_vs_rag.config import AppConfig
from llm_wiki_vs_rag.rag.pipeline import _new_run_id as rag_run_id
from llm_wiki_vs_rag.wiki.pipeline import _new_run_id as wiki_run_id


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
                "system,query_id,phase,accuracy,synthesis,latest_state,contradiction_detected,contradiction_resolved,compression_loss,provenance_fidelity,evaluator_notes",
                "rag,q1,phase_1,correct,full,correct,true,true,none,true,looks good",
                "wiki,q2,phase_2,wrong,failed,stale,false,false,major,false,bad output",
            ]
        ),
        encoding="utf-8",
    )

    labels = load_manual_labels(labels_path)

    assert labels[("rag", "q1", "phase_1")].accuracy == "correct"
    assert labels[("rag", "q1", "phase_1")].contradiction_detected is True
    assert labels[("wiki", "q2", "phase_2")].compression_loss == "major"


def test_metric_aggregation_and_drift():
    labels_csv = "\n".join([
        "system,query_id,phase,accuracy,synthesis,latest_state,contradiction_detected,contradiction_resolved,compression_loss,provenance_fidelity,evaluator_notes",
        "rag,q1,phase_1,correct,full,correct,true,true,none,true,",
        "rag,q2,phase_2,wrong,failed,stale,false,false,major,false,",
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


def test_contradiction_resolved_pct_uses_only_detected_rows():
    labels_csv = "\n".join([
        "system,query_id,phase,accuracy,synthesis,latest_state,contradiction_detected,contradiction_resolved,compression_loss,provenance_fidelity,evaluator_notes",
        "rag,q1,phase_1,correct,full,correct,true,true,none,true,",
        "rag,q2,phase_1,correct,full,correct,false,false,none,true,",
    ])
    outputs = [
        RunOutputRecord(query_id="q1", system="rag", phase="phase_1", question="Q1", category="policy", answer="A1"),
        RunOutputRecord(query_id="q2", system="rag", phase="phase_1", question="Q2", category="policy", answer="A2"),
    ]
    from tempfile import NamedTemporaryFile
    with NamedTemporaryFile("w+", suffix=".csv") as tmp:
        tmp.write(labels_csv)
        tmp.flush()
        records = merge_outputs_with_labels(outputs, load_manual_labels(Path(tmp.name)))

    by_system = summarize_records(records, group_fields=("system",))
    assert by_system[0].metrics["contradiction"]["resolved_pct"] == 100.0


def test_contradiction_resolved_pct_is_not_applicable_when_no_detected_rows():
    labels_csv = "\n".join([
        "system,query_id,phase,accuracy,synthesis,latest_state,contradiction_detected,contradiction_resolved,compression_loss,provenance_fidelity,evaluator_notes",
        "rag,q1,phase_1,correct,full,correct,false,false,none,true,",
    ])
    outputs = [
        RunOutputRecord(query_id="q1", system="rag", phase="phase_1", question="Q1", category="policy", answer="A1"),
    ]
    from tempfile import NamedTemporaryFile
    with NamedTemporaryFile("w+", suffix=".csv") as tmp:
        tmp.write(labels_csv)
        tmp.flush()
        records = merge_outputs_with_labels(outputs, load_manual_labels(Path(tmp.name)))

    by_system = summarize_records(records, group_fields=("system",))
    assert by_system[0].metrics["contradiction"]["resolved_pct"] is None


def test_report_file_generation(tmp_path):
    labels_path = tmp_path / "labels.csv"
    with labels_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "query_id",
                "system",
                "phase",
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
        writer.writerow(["q1", "rag", "phase_1", "correct", "full", "correct", "true", "true", "none", "true", ""])
        writer.writerow(["q2", "rag", "phase_2", "wrong", "failed", "stale", "false", "false", "major", "false", ""])

    records = merge_outputs_with_labels(_sample_outputs(), load_manual_labels(labels_path))
    output_dir = tmp_path / "report"
    write_reports(records, output_dir)

    assert (output_dir / "summary.json").exists()
    assert (output_dir / "summary.csv").exists()
    assert (output_dir / "per_query_results.csv").exists()
    assert (output_dir / "report.md").exists()

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert "summaries_by_system" in summary

def test_report_outputs_na_for_contradiction_pct_when_not_applicable(tmp_path):
    labels_path = tmp_path / "labels.csv"
    labels_path.write_text(
        "\n".join(
            [
                "system,query_id,phase,accuracy,synthesis,latest_state,contradiction_detected,contradiction_resolved,compression_loss,provenance_fidelity,evaluator_notes",
                "rag,q1,phase_1,correct,full,correct,false,false,none,true,",
            ]
        ),
        encoding="utf-8",
    )
    records = merge_outputs_with_labels(
        [
            RunOutputRecord(
                query_id="q1",
                system="rag",
                phase="phase_1",
                question="Q1",
                category="policy",
                answer="A1",
            )
        ],
        load_manual_labels(labels_path),
    )
    output_dir = tmp_path / "report"
    write_reports(records, output_dir)

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["summaries_by_system"][0]["metrics"]["contradiction"]["resolved_pct"] is None
    assert "N/A" in (output_dir / "summary.csv").read_text(encoding="utf-8")
    assert "N/A" in (output_dir / "report.md").read_text(encoding="utf-8")


def test_merge_uses_system_plus_query_id_plus_phase_for_labels():
    outputs = [
        RunOutputRecord(query_id="q1", system="rag", phase="phase_1", question="Q", category="policy", answer="A"),
        RunOutputRecord(query_id="q1", system="wiki", phase="phase_1", question="Q", category="policy", answer="A"),
    ]
    labels = {
        ("rag", "q1", "phase_1"): load_manual_labels_from_row("rag", "q1", "phase_1", "correct"),
        ("wiki", "q1", "phase_1"): load_manual_labels_from_row("wiki", "q1", "phase_1", "wrong"),
    }

    records = merge_outputs_with_labels(outputs, labels)
    by_system = {record.system: record for record in records}
    assert by_system["rag"].accuracy == "correct"
    assert by_system["wiki"].accuracy == "wrong"


def test_merge_fails_when_any_output_row_is_unlabeled():
    outputs = [
        RunOutputRecord(query_id="q1", system="rag", phase="phase_1", question="Q1", category="policy", answer="A1"),
        RunOutputRecord(query_id="q2", system="rag", phase="phase_2", question="Q2", category="policy", answer="A2"),
    ]
    labels = {
        ("rag", "q1", "phase_1"): load_manual_labels_from_row("rag", "q1", "phase_1", "correct"),
    }

    try:
        merge_outputs_with_labels(outputs, labels)
    except ValueError as exc:
        assert "missing_or_partial_label_sample" in str(exc)
        assert "q2" in str(exc)
        assert "phase_2" in str(exc)
    else:
        raise AssertionError("Expected unlabeled output rows to fail evaluation merge.")


def test_merge_fails_when_any_output_row_has_partial_label():
    outputs = [
        RunOutputRecord(query_id="q1", system="rag", phase="phase_1", question="Q1", category="policy", answer="A1"),
    ]
    partial_label = load_manual_labels_from_row("rag", "q1", "phase_1", "correct").model_copy(
        update={"accuracy": None}
    )
    labels = {("rag", "q1", "phase_1"): partial_label}

    try:
        merge_outputs_with_labels(outputs, labels)
    except ValueError as exc:
        assert "missing_or_partial_label_sample" in str(exc)
        assert "q1" in str(exc)
        assert "phase_1" in str(exc)
    else:
        raise AssertionError("Expected partial label rows to fail evaluation merge.")


def load_manual_labels_from_row(system: str, query_id: str, phase: str, accuracy: str):
    return load_manual_labels_from_text(
        "\n".join(
            [
                "system,query_id,phase,accuracy,synthesis,latest_state,contradiction_detected,contradiction_resolved,compression_loss,provenance_fidelity,evaluator_notes",
                f"{system},{query_id},{phase},{accuracy},full,correct,true,true,none,true,",
            ]
        )
    )[(system, query_id, phase)]


def load_manual_labels_from_text(csv_payload: str):
    from tempfile import NamedTemporaryFile

    with NamedTemporaryFile("w+", suffix=".csv") as tmp:
        tmp.write(csv_payload)
        tmp.flush()
        return load_manual_labels(Path(tmp.name))


def test_run_queries_for_system_preserves_per_query_latency_and_run_id(monkeypatch, tmp_path):
    def _fake_rag(**_kwargs):
        return [
            GenerationResult(query_id="q1", answer="a1", mode="rag", run_id="r1", latency_ms=11.1, artifact_dir="art/1", prompt_tokens=3, completion_tokens=4, total_tokens=7),
            GenerationResult(query_id="q2", answer="a2", mode="rag", run_id="r2", latency_ms=22.2, artifact_dir="art/2", prompt_tokens=5, completion_tokens=6, total_tokens=11),
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
    assert [record.prompt_tokens for record in records] == [3, 5]
    assert [record.completion_tokens for record in records] == [4, 6]
    assert [record.total_tokens for record in records] == [7, 11]
    assert all(record.metadata.get("corpus_snapshot") for record in records)


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
            prompt_tokens=12,
            completion_tokens=8,
            total_tokens=20,
            metadata={"artifact_dir": "artifacts/rag_runs/rag-1"},
        )
    ]
    labels = {("rag", "q1", "phase_1"): load_manual_labels_from_row("rag", "q1", "phase_1", "correct")}
    output_dir = tmp_path / "report_traceability"
    write_reports(merge_outputs_with_labels(records, labels), output_dir)

    per_query = (output_dir / "per_query_results.csv").read_text(encoding="utf-8")
    assert "run_id" in per_query
    assert "artifact_dir" in per_query
    assert "rag-1" in per_query
    assert "prompt_tokens" in per_query
    assert "completion_tokens" in per_query
    assert ",20," in per_query


def test_wildcard_labels_are_rejected(tmp_path):
    labels_path = tmp_path / "labels.csv"
    labels_path.write_text(
        "\n".join(
            [
                "system,query_id,phase,accuracy,synthesis,latest_state,contradiction_detected,contradiction_resolved,compression_loss,provenance_fidelity,evaluator_notes",
                "*,q1,phase_1,correct,full,correct,true,true,none,true,",
            ]
        ),
        encoding="utf-8",
    )
    try:
        load_manual_labels(labels_path)
    except ValueError as exc:
        assert "wildcard labels are not supported" in str(exc)
    else:
        raise AssertionError("Expected wildcard label rows to be rejected.")


def test_same_query_id_across_phases_keeps_distinct_labels():
    outputs = [
        RunOutputRecord(query_id="q1", system="rag", phase="phase_1", question="Q", category="policy", answer="A1"),
        RunOutputRecord(query_id="q1", system="rag", phase="phase_2", question="Q", category="policy", answer="A2"),
    ]
    labels = {
        ("rag", "q1", "phase_1"): load_manual_labels_from_row("rag", "q1", "phase_1", "correct"),
        ("rag", "q1", "phase_2"): load_manual_labels_from_row("rag", "q1", "phase_2", "wrong"),
    }

    records = merge_outputs_with_labels(outputs, labels)
    by_phase = {record.phase: record for record in records}
    assert by_phase["phase_1"].accuracy == "correct"
    assert by_phase["phase_2"].accuracy == "wrong"


def test_drift_contradiction_resolved_uses_detected_denominator_only():
    outputs = [
        RunOutputRecord(query_id="q1", system="rag", phase="phase_1", question="Q1", category="policy", answer="A1"),
        RunOutputRecord(query_id="q2", system="rag", phase="phase_1", question="Q2", category="policy", answer="A2"),
        RunOutputRecord(query_id="q3", system="rag", phase="phase_2", question="Q3", category="policy", answer="A3"),
        RunOutputRecord(query_id="q4", system="rag", phase="phase_2", question="Q4", category="policy", answer="A4"),
    ]
    labels = load_manual_labels_from_text(
        "\n".join(
            [
                "system,query_id,phase,accuracy,synthesis,latest_state,contradiction_detected,contradiction_resolved,compression_loss,provenance_fidelity,evaluator_notes",
                "rag,q1,phase_1,correct,full,correct,true,true,none,true,",
                "rag,q2,phase_1,correct,full,correct,false,false,none,true,",
                "rag,q3,phase_2,correct,full,correct,true,false,none,true,",
                "rag,q4,phase_2,correct,full,correct,false,false,none,true,",
            ]
        )
    )
    records = merge_outputs_with_labels(outputs, labels)
    drifts = compute_drift(records)
    assert drifts[0].contradiction_resolved_rate_delta == -1.0


def test_load_manual_labels_fails_when_phase_missing(tmp_path):
    labels_path = tmp_path / "labels.csv"
    labels_path.write_text(
        "\n".join(
            [
                "system,query_id,accuracy,synthesis,latest_state,contradiction_detected,contradiction_resolved,compression_loss,provenance_fidelity,evaluator_notes",
                "rag,q1,correct,full,correct,true,true,none,true,",
            ]
        ),
        encoding="utf-8",
    )

    try:
        load_manual_labels(labels_path)
    except ValueError as exc:
        assert "must include a phase value" in str(exc)
    else:
        raise AssertionError("Expected labels without phase to fail.")


def test_repeated_runs_within_same_second_have_distinct_run_ids():
    rag_first = rag_run_id("q1")
    rag_second = rag_run_id("q1")
    wiki_first = wiki_run_id("q1")
    wiki_second = wiki_run_id("q1")

    assert rag_first != rag_second
    assert wiki_first != wiki_second


def test_rag_snapshot_resolver_uses_canonical_manifest_path(tmp_path):
    paths = ProjectPaths(project_root=tmp_path)
    paths.ensure()
    manifest_path = paths.artifacts_dir / "rag_index" / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text('{"snapshot_id": "rag-snapshot-001"}', encoding="utf-8")

    snapshot = resolve_corpus_snapshot_identity(paths=paths, system="rag")
    assert snapshot == "rag-snapshot-001"


def test_rag_snapshot_resolver_ignores_legacy_manifest_locations(tmp_path):
    paths = ProjectPaths(project_root=tmp_path)
    paths.ensure()
    (paths.artifacts_dir / "manifest.json").write_text('{"snapshot_id": "legacy-root"}', encoding="utf-8")
    (paths.artifacts_dir / "rag_index.manifest.json").write_text('{"snapshot_id": "legacy-rag"}', encoding="utf-8")

    snapshot = resolve_corpus_snapshot_identity(paths=paths, system="rag")
    assert snapshot == str((paths.artifacts_dir / "rag_index" / "manifest.json").resolve())
