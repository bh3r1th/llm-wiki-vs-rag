from __future__ import annotations

import json
from pathlib import Path

from llm_wiki_vs_rag.config import AppConfig
from llm_wiki_vs_rag.runner import run_command


def _write_run_file(path: Path, system: str, query_ids: tuple[str, ...]) -> None:
    rows = []
    for query_id in query_ids:
        rows.append(
            {
                "query_id": query_id,
                "system": system,
                "phase": "phase_1",
                "question": f"Question {query_id}",
                "category": "policy",
                "answer": f"Answer {query_id}",
                "metadata": {
                    "corpus_snapshot": "snapshot-a",
                    "corpus_order": "001",
                    "execution_fingerprint": f"exec-{system}",
                },
            }
        )
        rows.append(
            {
                "query_id": query_id,
                "system": system,
                "phase": "phase_2",
                "question": f"Question {query_id}",
                "category": "policy",
                "answer": f"Answer {query_id}",
                "metadata": {
                    "corpus_snapshot": "snapshot-b",
                    "corpus_order": "002",
                    "execution_fingerprint": f"exec-{system}",
                },
            }
        )
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _write_labels(path: Path, query_ids: tuple[str, ...], include_wiki: bool = True) -> None:
    lines = [
        "system,query_id,phase,accuracy,synthesis,latest_state,contradiction_detected,contradiction_resolved,compression_loss,provenance_fidelity,evaluator_notes"
    ]
    for query_id in query_ids:
        for phase in ("phase_1", "phase_2"):
            lines.append(f"rag,{query_id},{phase},correct,full,correct,true,true,none,true,")
            if include_wiki:
                lines.append(f"wiki,{query_id},{phase},correct,full,correct,true,true,none,true,")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_smoke_evaluate_runs_all_three_paths_successfully(tmp_path):
    rag_run_file = tmp_path / "rag.jsonl"
    wiki_run_file = tmp_path / "wiki.jsonl"
    labels_file = tmp_path / "labels.csv"
    output_dir = tmp_path / "smoke-eval"
    _write_run_file(rag_run_file, "rag", ("q1",))
    _write_run_file(wiki_run_file, "wiki", ("q1",))
    _write_labels(labels_file, ("q1",))

    run_command(
        "smoke-evaluate",
        AppConfig(project_root=tmp_path),
        rag_run_file=str(rag_run_file),
        wiki_run_file=str(wiki_run_file),
        labels_file=str(labels_file),
        output_dir=str(output_dir),
    )

    assert (output_dir / "rag" / "summary.json").exists()
    assert (output_dir / "wiki" / "summary.json").exists()
    assert (output_dir / "compare" / "summary.json").exists()


def test_smoke_evaluate_fails_early_when_labels_missing(tmp_path):
    rag_run_file = tmp_path / "rag.jsonl"
    wiki_run_file = tmp_path / "wiki.jsonl"
    labels_file = tmp_path / "labels.csv"
    output_dir = tmp_path / "smoke-eval"
    _write_run_file(rag_run_file, "rag", ("q1",))
    _write_run_file(wiki_run_file, "wiki", ("q1",))
    _write_labels(labels_file, ("q1",), include_wiki=False)

    try:
        run_command(
            "smoke-evaluate",
            AppConfig(project_root=tmp_path),
            rag_run_file=str(rag_run_file),
            wiki_run_file=str(wiki_run_file),
            labels_file=str(labels_file),
            output_dir=str(output_dir),
        )
    except ValueError as exc:
        assert "missing_or_partial_label_sample" in str(exc)
        assert "wiki" in str(exc)
    else:
        raise AssertionError("Expected smoke-evaluate to fail before evaluation with missing manual labels.")

    assert not (output_dir / "rag").exists()
    assert not (output_dir / "wiki").exists()
    assert not (output_dir / "compare").exists()


def test_smoke_evaluate_fails_early_on_bad_cross_system_cohort_alignment(tmp_path):
    rag_run_file = tmp_path / "rag.jsonl"
    wiki_run_file = tmp_path / "wiki.jsonl"
    labels_file = tmp_path / "labels.csv"
    output_dir = tmp_path / "smoke-eval"
    _write_run_file(rag_run_file, "rag", ("q1",))
    _write_run_file(wiki_run_file, "wiki", ("q2",))
    _write_labels(labels_file, ("q1", "q2"))

    try:
        run_command(
            "smoke-evaluate",
            AppConfig(project_root=tmp_path),
            rag_run_file=str(rag_run_file),
            wiki_run_file=str(wiki_run_file),
            labels_file=str(labels_file),
            output_dir=str(output_dir),
        )
    except ValueError as exc:
        assert "mismatched (query_id, phase) cohorts" in str(exc)
    else:
        raise AssertionError("Expected smoke-evaluate to fail before evaluation on cohort mismatch.")

    assert not (output_dir / "rag").exists()
    assert not (output_dir / "wiki").exists()
    assert not (output_dir / "compare").exists()


def test_smoke_evaluate_creates_expected_output_dir_structure(tmp_path):
    rag_run_file = tmp_path / "rag.jsonl"
    wiki_run_file = tmp_path / "wiki.jsonl"
    labels_file = tmp_path / "labels.csv"
    output_dir = tmp_path / "smoke-eval"
    _write_run_file(rag_run_file, "rag", ("q1", "q2"))
    _write_run_file(wiki_run_file, "wiki", ("q1", "q2"))
    _write_labels(labels_file, ("q1", "q2"))

    run_command(
        "smoke-evaluate",
        AppConfig(project_root=tmp_path),
        rag_run_file=str(rag_run_file),
        wiki_run_file=str(wiki_run_file),
        labels_file=str(labels_file),
        output_dir=str(output_dir),
    )

    assert (output_dir / "rag").is_dir()
    assert (output_dir / "wiki").is_dir()
    assert (output_dir / "compare").is_dir()
