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
from llm_wiki_vs_rag.eval.models import EvalQueryCase, RunOutputRecord
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


def test_load_manual_labels_rejects_resolved_without_detected(tmp_path):
    labels_path = tmp_path / "labels.csv"
    labels_path.write_text(
        "\n".join(
            [
                "system,query_id,phase,accuracy,synthesis,latest_state,contradiction_detected,contradiction_resolved,compression_loss,provenance_fidelity,evaluator_notes",
                "rag,q1,phase_1,correct,full,correct,false,true,none,true,invalid",
            ]
        ),
        encoding="utf-8",
    )
    try:
        load_manual_labels(labels_path)
    except ValueError as exc:
        assert "Invalid contradiction labels" in str(exc)
        assert "q1" in str(exc)
        assert "phase_1" in str(exc)
    else:
        raise AssertionError("Expected contradiction_resolved=True without detected to fail.")


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


def test_contradiction_resolved_percentage_cannot_exceed_100():
    outputs = [
        RunOutputRecord(query_id="q1", system="rag", phase="phase_1", question="Q1", category="policy", answer="A1"),
    ]
    labels = {
        (
            "rag",
            "q1",
            "phase_1",
        ): load_manual_labels_from_row("rag", "q1", "phase_1", "correct").model_copy(
            update={"contradiction_detected": False, "contradiction_resolved": True}
        )
    }
    try:
        merge_outputs_with_labels(outputs, labels)
    except ValueError as exc:
        assert "Invalid contradiction labels" in str(exc)
    else:
        raise AssertionError("Expected invalid contradiction labels to fail before percentage computation.")


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
    artifact_one = tmp_path / "art" / "1"
    artifact_two = tmp_path / "art" / "2"
    artifact_one.mkdir(parents=True, exist_ok=True)
    artifact_two.mkdir(parents=True, exist_ok=True)
    (artifact_one / "metadata.json").write_text(json.dumps({"mode": "rag", "corpus_snapshot": "sha256:rag-snapshot"}), encoding="utf-8")
    (artifact_two / "metadata.json").write_text(json.dumps({"mode": "rag", "corpus_snapshot": "sha256:rag-snapshot"}), encoding="utf-8")

    def _fake_rag(**_kwargs):
        return [
            GenerationResult(query_id="q1::phase=phase_1", answer="a1", mode="rag", run_id="r1", latency_ms=11.1, artifact_dir=str(artifact_one), prompt_tokens=3, completion_tokens=4, total_tokens=7),
            GenerationResult(query_id="q2::phase=phase_1", answer="a2", mode="rag", run_id="r2", latency_ms=22.2, artifact_dir=str(artifact_two), prompt_tokens=5, completion_tokens=6, total_tokens=11),
        ]

    monkeypatch.setattr("llm_wiki_vs_rag.eval.harness.run_rag_queries", _fake_rag)
    paths = ProjectPaths(project_root=tmp_path)
    paths.ensure()
    (paths.artifacts_dir / "rag_index").mkdir(parents=True, exist_ok=True)
    (paths.artifacts_dir / "rag_index" / "manifest.json").write_text(
        json.dumps({"snapshot_id": "sha256:rag-snapshot"}),
        encoding="utf-8",
    )
    records = run_queries_for_system(
        config=AppConfig(project_root=tmp_path),
        paths=paths,
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
    assert all(record.metadata.get("corpus_snapshot") == "sha256:rag-snapshot" for record in records)


def test_run_queries_for_system_wiki_records_written_snapshot_identity(monkeypatch, tmp_path):
    artifact_dir = tmp_path / "art" / "w1"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "metadata.json").write_text(json.dumps({"mode": "wiki", "corpus_snapshot": "sha256:wiki-snapshot"}), encoding="utf-8")
    monkeypatch.setattr(
        "llm_wiki_vs_rag.eval.harness.run_wiki_queries",
        lambda **_kwargs: [
            GenerationResult(query_id="q1::phase=phase_1", answer="a1", mode="wiki", run_id="w1", latency_ms=1.1, artifact_dir=str(artifact_dir))
        ],
    )
    paths = ProjectPaths(project_root=tmp_path)
    paths.ensure()
    (paths.wiki_dir / "snapshot.json").write_text(json.dumps({"snapshot_id": "sha256:wiki-snapshot"}), encoding="utf-8")

    records = run_queries_for_system(
        config=AppConfig(project_root=tmp_path),
        paths=paths,
        query_cases=[load_query_case("q1", "Q1")],
        system="wiki",
    )
    assert records[0].metadata.get("corpus_snapshot") == "sha256:wiki-snapshot"


def test_run_queries_for_system_derives_snapshot_from_runtime_manifest(monkeypatch, tmp_path):
    artifact_dir = tmp_path / "art" / "r1"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "metadata.json").write_text(
        json.dumps({"mode": "rag", "corpus_snapshot": "sha256:runtime-manifest"}),
        encoding="utf-8",
    )
    captured: dict[str, str | None] = {}

    def _fake_rag(**kwargs):
        captured["corpus_snapshot"] = kwargs.get("corpus_snapshot")
        return [
            GenerationResult(query_id="q1::phase=phase_1", answer="a1", mode="rag", run_id="r1", artifact_dir=str(artifact_dir))
        ]

    monkeypatch.setattr("llm_wiki_vs_rag.eval.harness.run_rag_queries", _fake_rag)
    paths = ProjectPaths(project_root=tmp_path)
    paths.ensure()
    (paths.artifacts_dir / "rag_index").mkdir(parents=True, exist_ok=True)
    (paths.artifacts_dir / "rag_index" / "manifest.json").write_text(
        json.dumps({"snapshot_id": "sha256:runtime-manifest"}),
        encoding="utf-8",
    )

    records = run_queries_for_system(
        config=AppConfig(project_root=tmp_path),
        paths=paths,
        query_cases=[load_query_case("q1", "Q1")],
        system="rag",
    )

    assert captured["corpus_snapshot"] == "sha256:runtime-manifest"
    assert records[0].metadata.get("corpus_snapshot") == "sha256:runtime-manifest"


def test_run_queries_for_system_keeps_phase_identity_when_query_id_repeats(monkeypatch, tmp_path):
    artifact_one = tmp_path / "art" / "repeat-1"
    artifact_two = tmp_path / "art" / "repeat-2"
    artifact_one.mkdir(parents=True, exist_ok=True)
    artifact_two.mkdir(parents=True, exist_ok=True)
    (artifact_one / "metadata.json").write_text(json.dumps({"mode": "rag", "corpus_snapshot": "sha256:rag-snapshot"}), encoding="utf-8")
    (artifact_two / "metadata.json").write_text(json.dumps({"mode": "rag", "corpus_snapshot": "sha256:rag-snapshot"}), encoding="utf-8")
    monkeypatch.setattr(
        "llm_wiki_vs_rag.eval.harness.run_rag_queries",
        lambda **_kwargs: [
            GenerationResult(query_id="q1::phase=phase_1", answer="a1", mode="rag", artifact_dir=str(artifact_one)),
            GenerationResult(query_id="q1::phase=phase_2", answer="a2", mode="rag", artifact_dir=str(artifact_two)),
        ],
    )
    paths = ProjectPaths(project_root=tmp_path)
    paths.ensure()
    (paths.artifacts_dir / "rag_index").mkdir(parents=True, exist_ok=True)
    (paths.artifacts_dir / "rag_index" / "manifest.json").write_text(
        json.dumps({"snapshot_id": "sha256:rag-snapshot"}),
        encoding="utf-8",
    )
    records = run_queries_for_system(
        config=AppConfig(project_root=tmp_path),
        paths=paths,
        query_cases=[
            EvalQueryCase(query_id="q1", question="Phase one question", category="policy", phase="phase_1"),
            EvalQueryCase(query_id="q1", question="Phase two question", category="history", phase="phase_2"),
        ],
        system="rag",
    )

    assert [record.phase for record in records] == ["phase_1", "phase_2"]
    assert [record.question for record in records] == ["Phase one question", "Phase two question"]
    assert [record.category for record in records] == ["policy", "history"]


def test_run_queries_for_system_fails_on_mixed_phases_without_explicit_phase_binding(tmp_path):
    paths = ProjectPaths(project_root=tmp_path)
    paths.ensure()
    (paths.artifacts_dir / "rag_index").mkdir(parents=True, exist_ok=True)
    (paths.artifacts_dir / "rag_index" / "manifest.json").write_text(
        json.dumps({"snapshot_id": "sha256:canonical-snapshot"}),
        encoding="utf-8",
    )
    try:
        run_queries_for_system(
            config=AppConfig(project_root=tmp_path),
            paths=paths,
            query_cases=[
                EvalQueryCase(query_id="q1", question="Q1", category="policy", phase="phase_1"),
                EvalQueryCase(query_id="q2", question="Q2", category="policy", phase="phase_2"),
            ],
            system="rag",
        )
    except ValueError as exc:
        assert "Mixed-phase query execution is not allowed without explicit phase binding" in str(exc)
    else:
        raise AssertionError("Expected mixed-phase query invocation without explicit phase binding to fail.")


def test_run_queries_for_system_accepts_explicit_phase_binding_with_runtime_snapshot(monkeypatch, tmp_path):
    artifact_dir = tmp_path / "artifacts" / "rag_runs" / "run-1"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "metadata.json").write_text(json.dumps({"mode": "rag", "corpus_snapshot": "sha256:phase-2-snapshot"}), encoding="utf-8")
    monkeypatch.setattr(
        "llm_wiki_vs_rag.eval.harness.run_rag_queries",
        lambda **_kwargs: [
            GenerationResult(query_id="q2::phase=phase_2", answer="a2", mode="rag", artifact_dir=str(artifact_dir)),
        ],
    )
    paths = ProjectPaths(project_root=tmp_path)
    paths.ensure()
    (paths.artifacts_dir / "rag_index").mkdir(parents=True, exist_ok=True)
    (paths.artifacts_dir / "rag_index" / "manifest.json").write_text(
        json.dumps({"snapshot_id": "sha256:phase-2-snapshot"}),
        encoding="utf-8",
    )
    records = run_queries_for_system(
        config=AppConfig(project_root=tmp_path),
        paths=paths,
        query_cases=[
            EvalQueryCase(query_id="q1", question="Q1", category="policy", phase="phase_1"),
            EvalQueryCase(query_id="q2", question="Q2", category="history", phase="phase_2"),
        ],
        system="rag",
        target_phase="phase_2",
    )
    assert len(records) == 1
    assert records[0].phase == "phase_2"
    assert records[0].metadata.get("corpus_snapshot") == "sha256:phase-2-snapshot"


def test_run_queries_normalization_is_order_independent(monkeypatch, tmp_path):
    artifact_one = tmp_path / "art" / "reordered-1"
    artifact_two = tmp_path / "art" / "reordered-2"
    artifact_one.mkdir(parents=True, exist_ok=True)
    artifact_two.mkdir(parents=True, exist_ok=True)
    (artifact_one / "metadata.json").write_text(json.dumps({"mode": "rag", "corpus_snapshot": "sha256:rag-snapshot"}), encoding="utf-8")
    (artifact_two / "metadata.json").write_text(json.dumps({"mode": "rag", "corpus_snapshot": "sha256:rag-snapshot"}), encoding="utf-8")

    def _fake_rag(**_kwargs):
        return [
            GenerationResult(query_id="q1::phase=phase_2", answer="a2", mode="rag", artifact_dir=str(artifact_two)),
            GenerationResult(query_id="q1::phase=phase_1", answer="a1", mode="rag", artifact_dir=str(artifact_one)),
        ]

    monkeypatch.setattr("llm_wiki_vs_rag.eval.harness.run_rag_queries", _fake_rag)
    paths = ProjectPaths(project_root=tmp_path)
    paths.ensure()
    (paths.artifacts_dir / "rag_index").mkdir(parents=True, exist_ok=True)
    (paths.artifacts_dir / "rag_index" / "manifest.json").write_text(
        json.dumps({"snapshot_id": "sha256:rag-snapshot"}),
        encoding="utf-8",
    )

    records = run_queries_for_system(
        config=AppConfig(project_root=tmp_path),
        paths=paths,
        query_cases=[
            EvalQueryCase(query_id="q1", question="Phase one question", category="policy", phase="phase_1"),
            EvalQueryCase(query_id="q1", question="Phase two question", category="history", phase="phase_2"),
        ],
        system="rag",
    )
    by_phase = {record.phase: record for record in records}
    assert by_phase["phase_1"].question == "Phase one question"
    assert by_phase["phase_1"].category == "policy"
    assert by_phase["phase_2"].question == "Phase two question"
    assert by_phase["phase_2"].category == "history"


def test_run_queries_for_system_fails_on_snapshot_mismatch_in_artifact_metadata(monkeypatch, tmp_path):
    artifact_dir = tmp_path / "artifacts" / "rag_runs" / "run-1"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "metadata.json").write_text(json.dumps({"mode": "rag", "corpus_snapshot": "sha256:stale-snapshot"}), encoding="utf-8")
    monkeypatch.setattr(
        "llm_wiki_vs_rag.eval.harness.run_rag_queries",
        lambda **_kwargs: [
            GenerationResult(query_id="q1::phase=phase_1", answer="a1", mode="rag", artifact_dir=str(artifact_dir)),
        ],
    )
    paths = ProjectPaths(project_root=tmp_path)
    paths.ensure()
    (paths.artifacts_dir / "rag_index").mkdir(parents=True, exist_ok=True)
    (paths.artifacts_dir / "rag_index" / "manifest.json").write_text(
        json.dumps({"snapshot_id": "sha256:canonical-snapshot"}),
        encoding="utf-8",
    )

    try:
        run_queries_for_system(
            config=AppConfig(project_root=tmp_path),
            paths=paths,
            query_cases=[load_query_case("q1", "Q1")],
            system="rag",
        )
    except ValueError as exc:
        assert "Run snapshot attribution mismatch" in str(exc)
    else:
        raise AssertionError("Expected run snapshot attribution mismatch to fail fast.")


def test_run_queries_for_system_fails_when_artifact_snapshot_missing(monkeypatch, tmp_path):
    artifact_dir = tmp_path / "artifacts" / "rag_runs" / "run-1"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "metadata.json").write_text(json.dumps({"mode": "rag", "query_id": "q1"}), encoding="utf-8")
    monkeypatch.setattr(
        "llm_wiki_vs_rag.eval.harness.run_rag_queries",
        lambda **_kwargs: [
            GenerationResult(query_id="q1::phase=phase_1", answer="a1", mode="rag", artifact_dir=str(artifact_dir)),
        ],
    )
    paths = ProjectPaths(project_root=tmp_path)
    paths.ensure()
    (paths.artifacts_dir / "rag_index").mkdir(parents=True, exist_ok=True)
    (paths.artifacts_dir / "rag_index" / "manifest.json").write_text(
        json.dumps({"snapshot_id": "sha256:canonical-snapshot"}),
        encoding="utf-8",
    )

    try:
        run_queries_for_system(
            config=AppConfig(project_root=tmp_path),
            paths=paths,
            query_cases=[load_query_case("q1", "Q1")],
            system="rag",
        )
    except ValueError as exc:
        assert "Missing corpus_snapshot in per-query artifact metadata" in str(exc)
    else:
        raise AssertionError("Expected missing artifact corpus_snapshot to fail fast.")


def test_run_queries_for_system_fails_on_result_mode_mismatch(monkeypatch, tmp_path):
    artifact_dir = tmp_path / "artifacts" / "rag_runs" / "run-1"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "metadata.json").write_text(json.dumps({"mode": "wiki", "corpus_snapshot": "sha256:canonical-snapshot"}), encoding="utf-8")
    monkeypatch.setattr(
        "llm_wiki_vs_rag.eval.harness.run_rag_queries",
        lambda **_kwargs: [
            GenerationResult(query_id="q1::phase=phase_1", answer="a1", mode="wiki", artifact_dir=str(artifact_dir)),
        ],
    )
    paths = ProjectPaths(project_root=tmp_path)
    paths.ensure()
    (paths.artifacts_dir / "rag_index").mkdir(parents=True, exist_ok=True)
    (paths.artifacts_dir / "rag_index" / "manifest.json").write_text(
        json.dumps({"snapshot_id": "sha256:canonical-snapshot"}),
        encoding="utf-8",
    )

    try:
        run_queries_for_system(
            config=AppConfig(project_root=tmp_path),
            paths=paths,
            query_cases=[load_query_case("q1", "Q1")],
            system="rag",
        )
    except ValueError as exc:
        assert "expected result mode rag, got wiki" in str(exc)
    else:
        raise AssertionError("Expected rag benchmark run to fail on mismatched result mode.")


def test_run_queries_for_system_fails_on_artifact_mode_mismatch(monkeypatch, tmp_path):
    artifact_dir = tmp_path / "artifacts" / "wiki_runs" / "run-1"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "metadata.json").write_text(json.dumps({"mode": "rag", "corpus_snapshot": "sha256:wiki-snapshot"}), encoding="utf-8")
    monkeypatch.setattr(
        "llm_wiki_vs_rag.eval.harness.run_wiki_queries",
        lambda **_kwargs: [
            GenerationResult(query_id="q1::phase=phase_1", answer="a1", mode="wiki", artifact_dir=str(artifact_dir)),
        ],
    )
    paths = ProjectPaths(project_root=tmp_path)
    paths.ensure()
    (paths.wiki_dir / "snapshot.json").write_text(json.dumps({"snapshot_id": "sha256:wiki-snapshot"}), encoding="utf-8")

    try:
        run_queries_for_system(
            config=AppConfig(project_root=tmp_path),
            paths=paths,
            query_cases=[load_query_case("q1", "Q1")],
            system="wiki",
        )
    except ValueError as exc:
        assert "expected artifact mode wiki, got rag" in str(exc)
    else:
        raise AssertionError("Expected wiki benchmark run to fail on mismatched artifact mode.")


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


def test_drift_fails_when_phase_cohorts_differ_by_query_identity_content():
    outputs = [
        RunOutputRecord(query_id="q1", system="rag", phase="phase_1", question="Q one", category="policy", answer="A1"),
        RunOutputRecord(query_id="q2", system="rag", phase="phase_2", question="Q two", category="policy", answer="A2"),
    ]
    labels = load_manual_labels_from_text(
        "\n".join(
            [
                "system,query_id,phase,accuracy,synthesis,latest_state,contradiction_detected,contradiction_resolved,compression_loss,provenance_fidelity,evaluator_notes",
                "rag,q1,phase_1,correct,full,correct,true,true,none,true,",
                "rag,q2,phase_2,correct,full,correct,true,true,none,true,",
            ]
        )
    )
    records = merge_outputs_with_labels(outputs, labels)
    try:
        compute_drift(records)
    except ValueError as exc:
        assert "identical phase_1 and phase_2 query cohorts" in str(exc)
        assert "q1" in str(exc)
        assert "q2" in str(exc)
    else:
        raise AssertionError("Expected drift computation to fail on mismatched phase cohorts.")


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


def test_load_manual_labels_rejects_duplicate_system_query_phase_rows(tmp_path):
    labels_path = tmp_path / "labels.csv"
    labels_path.write_text(
        "\n".join(
            [
                "system,query_id,phase,accuracy,synthesis,latest_state,contradiction_detected,contradiction_resolved,compression_loss,provenance_fidelity,evaluator_notes",
                "rag,q1,phase_1,correct,full,correct,true,true,none,true,",
                "rag,q1,phase_1,wrong,failed,stale,false,false,major,false,duplicate",
            ]
        ),
        encoding="utf-8",
    )

    try:
        load_manual_labels(labels_path)
    except ValueError as exc:
        assert "must be unique per (system, query_id, phase)" in str(exc)
        assert "q1" in str(exc)
        assert "phase_1" in str(exc)
    else:
        raise AssertionError("Expected duplicate manual labels to fail.")


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

    try:
        resolve_corpus_snapshot_identity(paths=paths, system="rag")
    except ValueError as exc:
        assert "Missing canonical snapshot manifest for rag" in str(exc)
    else:
        raise AssertionError("Expected snapshot resolver to fail when canonical rag snapshot manifest is missing.")


def test_wiki_snapshot_resolver_uses_canonical_manifest_path(tmp_path):
    paths = ProjectPaths(project_root=tmp_path)
    paths.ensure()
    (paths.wiki_dir / "snapshot.json").write_text('{"snapshot_id": "sha256:wiki-snapshot-001"}', encoding="utf-8")

    snapshot = resolve_corpus_snapshot_identity(paths=paths, system="wiki")
    assert snapshot == "sha256:wiki-snapshot-001"
