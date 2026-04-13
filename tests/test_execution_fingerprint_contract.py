from __future__ import annotations

import json

from llm_wiki_vs_rag.config import AppConfig
from llm_wiki_vs_rag.eval.harness import load_run_outputs
from llm_wiki_vs_rag.runner import run_command


def test_load_run_outputs_rejects_missing_execution_fingerprint(tmp_path):
    run_file = tmp_path / "outputs.jsonl"
    run_file.write_text(
        json.dumps(
            {
                "query_id": "q1",
                "system": "rag",
                "phase": "phase_1",
                "question": "Q1",
                "category": "policy",
                "answer": "A1",
                "metadata": {"corpus_snapshot": "snap-a", "corpus_order": "001"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    try:
        load_run_outputs(run_file)
    except ValueError as exc:
        assert "metadata.execution_fingerprint" in str(exc)
        assert "q1" in str(exc)
    else:
        raise AssertionError("Expected load_run_outputs to reject rows missing execution_fingerprint.")


def test_evaluate_rag_rejects_mixed_execution_fingerprint_within_phase(tmp_path):
    run_file = tmp_path / "rag.jsonl"
    run_file.write_text(
        json.dumps(
            {
                "query_id": "q1",
                "system": "rag",
                "phase": "phase_1",
                "question": "Q1",
                "category": "policy",
                "answer": "A1",
                "metadata": {
                    "corpus_snapshot": "snap-a",
                    "corpus_order": "001",
                    "execution_fingerprint": "sha256:exec-a",
                },
            }
        )
        + "\n"
        + json.dumps(
            {
                "query_id": "q2",
                "system": "rag",
                "phase": "phase_1",
                "question": "Q2",
                "category": "policy",
                "answer": "A2",
                "metadata": {
                    "corpus_snapshot": "snap-a",
                    "corpus_order": "001",
                    "execution_fingerprint": "sha256:exec-b",
                },
            }
        )
        + "\n"
        + json.dumps(
            {
                "query_id": "q3",
                "system": "rag",
                "phase": "phase_2",
                "question": "Q3",
                "category": "policy",
                "answer": "A3",
                "metadata": {
                    "corpus_snapshot": "snap-b",
                    "corpus_order": "002",
                    "execution_fingerprint": "sha256:exec-c",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    labels_file = tmp_path / "labels.csv"
    labels_file.write_text(
        "system,query_id,phase,accuracy,synthesis,latest_state,contradiction_detected,contradiction_resolved,compression_loss,provenance_fidelity,evaluator_notes\n",
        encoding="utf-8",
    )

    try:
        run_command("evaluate-rag", AppConfig(project_root=tmp_path), run_file=str(run_file), labels_file=str(labels_file))
    except ValueError as exc:
        assert "Inconsistent execution_fingerprint within system/phase cohort" in str(exc)
        assert "phase_1" in str(exc)
    else:
        raise AssertionError("Expected evaluate-rag to reject mixed execution_fingerprint within phase.")


def test_evaluate_wiki_rejects_mixed_execution_fingerprint_within_phase(tmp_path):
    run_file = tmp_path / "wiki.jsonl"
    run_file.write_text(
        json.dumps(
            {
                "query_id": "q1",
                "system": "wiki",
                "phase": "phase_1",
                "question": "Q1",
                "category": "policy",
                "answer": "A1",
                "metadata": {
                    "corpus_snapshot": "snap-a",
                    "corpus_order": "001",
                    "execution_fingerprint": "sha256:exec-a",
                },
            }
        )
        + "\n"
        + json.dumps(
            {
                "query_id": "q2",
                "system": "wiki",
                "phase": "phase_1",
                "question": "Q2",
                "category": "policy",
                "answer": "A2",
                "metadata": {
                    "corpus_snapshot": "snap-a",
                    "corpus_order": "001",
                    "execution_fingerprint": "sha256:exec-b",
                },
            }
        )
        + "\n"
        + json.dumps(
            {
                "query_id": "q3",
                "system": "wiki",
                "phase": "phase_2",
                "question": "Q3",
                "category": "policy",
                "answer": "A3",
                "metadata": {
                    "corpus_snapshot": "snap-b",
                    "corpus_order": "002",
                    "execution_fingerprint": "sha256:exec-c",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    labels_file = tmp_path / "labels.csv"
    labels_file.write_text(
        "system,query_id,phase,accuracy,synthesis,latest_state,contradiction_detected,contradiction_resolved,compression_loss,provenance_fidelity,evaluator_notes\n",
        encoding="utf-8",
    )

    try:
        run_command("evaluate-wiki", AppConfig(project_root=tmp_path), run_file=str(run_file), labels_file=str(labels_file))
    except ValueError as exc:
        assert "Inconsistent execution_fingerprint within system/phase cohort" in str(exc)
        assert "phase_1" in str(exc)
    else:
        raise AssertionError("Expected evaluate-wiki to reject mixed execution_fingerprint within phase.")


def test_compare_systems_rejects_inconsistent_execution_fingerprint_within_system_phase(tmp_path):
    rag_run_file = tmp_path / "rag.jsonl"
    rag_run_file.write_text(
        json.dumps(
            {
                "query_id": "q1",
                "system": "rag",
                "phase": "phase_1",
                "question": "Q1",
                "category": "policy",
                "answer": "A1",
                "metadata": {
                    "corpus_snapshot": "snap-a",
                    "corpus_order": "001",
                    "execution_fingerprint": "sha256:exec-rag-a",
                },
            }
        )
        + "\n"
        + json.dumps(
            {
                "query_id": "q2",
                "system": "rag",
                "phase": "phase_1",
                "question": "Q2",
                "category": "policy",
                "answer": "A2",
                "metadata": {
                    "corpus_snapshot": "snap-a",
                    "corpus_order": "001",
                    "execution_fingerprint": "sha256:exec-rag-b",
                },
            }
        )
        + "\n"
        + json.dumps(
            {
                "query_id": "q3",
                "system": "rag",
                "phase": "phase_2",
                "question": "Q3",
                "category": "policy",
                "answer": "A3",
                "metadata": {
                    "corpus_snapshot": "snap-b",
                    "corpus_order": "002",
                    "execution_fingerprint": "sha256:exec-rag-c",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    wiki_run_file = tmp_path / "wiki.jsonl"
    wiki_run_file.write_text(
        json.dumps(
            {
                "query_id": "q1",
                "system": "wiki",
                "phase": "phase_1",
                "question": "Q1",
                "category": "policy",
                "answer": "A1",
                "metadata": {
                    "corpus_snapshot": "snap-a",
                    "corpus_order": "001",
                    "execution_fingerprint": "sha256:exec-wiki-a",
                },
            }
        )
        + "\n"
        + json.dumps(
            {
                "query_id": "q2",
                "system": "wiki",
                "phase": "phase_1",
                "question": "Q2",
                "category": "policy",
                "answer": "A2",
                "metadata": {
                    "corpus_snapshot": "snap-a",
                    "corpus_order": "001",
                    "execution_fingerprint": "sha256:exec-wiki-a",
                },
            }
        )
        + "\n"
        + json.dumps(
            {
                "query_id": "q3",
                "system": "wiki",
                "phase": "phase_2",
                "question": "Q3",
                "category": "policy",
                "answer": "A3",
                "metadata": {
                    "corpus_snapshot": "snap-b",
                    "corpus_order": "002",
                    "execution_fingerprint": "sha256:exec-wiki-b",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    labels_file = tmp_path / "labels.csv"
    labels_file.write_text(
        "system,query_id,phase,accuracy,synthesis,latest_state,contradiction_detected,contradiction_resolved,compression_loss,provenance_fidelity,evaluator_notes\n",
        encoding="utf-8",
    )

    try:
        run_command(
            "compare-systems",
            AppConfig(project_root=tmp_path),
            rag_run_file=str(rag_run_file),
            wiki_run_file=str(wiki_run_file),
            labels_file=str(labels_file),
        )
    except ValueError as exc:
        assert "Inconsistent execution_fingerprint within system/phase cohort" in str(exc)
        assert "rag" in str(exc)
    else:
        raise AssertionError("Expected compare-systems to reject inconsistent execution_fingerprint per cohort.")
