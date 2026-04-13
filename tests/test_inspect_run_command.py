from __future__ import annotations

import json

from llm_wiki_vs_rag.config import AppConfig
from llm_wiki_vs_rag.runner import run_command


def _write_run_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _row(*, query_id: str, system: str, phase: str, snapshot: str = "snap", fingerprint: str = "fp"):
    return {
        "query_id": query_id,
        "system": system,
        "phase": phase,
        "question": f"Question {query_id}",
        "category": "lookup",
        "answer": "A",
        "metadata": {
            "corpus_snapshot": snapshot,
            "execution_fingerprint": fingerprint,
            "artifact_dir": f"/tmp/{system}/{phase}/{query_id}",
        },
    }


def test_inspect_run_valid_smoke_run_passes(tmp_path, capsys):
    run_file = tmp_path / "run.jsonl"
    _write_run_jsonl(
        run_file,
        [
            _row(query_id="q1", system="rag", phase="phase_1", snapshot="snap-a", fingerprint="fp-rag-a"),
            _row(query_id="q2", system="rag", phase="phase_1", snapshot="snap-a", fingerprint="fp-rag-a"),
            _row(query_id="q1", system="rag", phase="phase_2", snapshot="snap-b", fingerprint="fp-rag-b"),
            _row(query_id="q2", system="rag", phase="phase_2", snapshot="snap-b", fingerprint="fp-rag-b"),
        ],
    )

    run_command("inspect-run", AppConfig(project_root=tmp_path), run_file=str(run_file))

    out = capsys.readouterr().out
    assert "total_rows=4" in out
    assert "duplicate_identity_count=0" in out


def test_inspect_run_duplicate_identity_fails(tmp_path):
    run_file = tmp_path / "run.jsonl"
    row = _row(query_id="q1", system="rag", phase="phase_1")
    _write_run_jsonl(run_file, [row, row])

    try:
        run_command("inspect-run", AppConfig(project_root=tmp_path), run_file=str(run_file))
    except ValueError as exc:
        assert "unique per (system, query_id, phase)" in str(exc)
    else:
        raise AssertionError("Expected duplicate (system, query_id, phase) rows to fail.")


def test_inspect_run_missing_corpus_snapshot_fails(tmp_path):
    run_file = tmp_path / "run.jsonl"
    row = _row(query_id="q1", system="rag", phase="phase_1")
    row["metadata"]["corpus_snapshot"] = ""
    _write_run_jsonl(run_file, [row])

    try:
        run_command("inspect-run", AppConfig(project_root=tmp_path), run_file=str(run_file))
    except ValueError as exc:
        assert "metadata.corpus_snapshot" in str(exc)
    else:
        raise AssertionError("Expected missing corpus_snapshot to fail.")


def test_inspect_run_missing_execution_fingerprint_fails(tmp_path):
    run_file = tmp_path / "run.jsonl"
    row = _row(query_id="q1", system="rag", phase="phase_1")
    row["metadata"]["execution_fingerprint"] = ""
    _write_run_jsonl(run_file, [row])

    try:
        run_command("inspect-run", AppConfig(project_root=tmp_path), run_file=str(run_file))
    except ValueError as exc:
        assert "metadata.execution_fingerprint" in str(exc)
    else:
        raise AssertionError("Expected missing execution_fingerprint to fail.")


def test_inspect_run_mixed_snapshot_or_fingerprint_in_same_cohort_fails(tmp_path):
    run_file = tmp_path / "run.jsonl"
    _write_run_jsonl(
        run_file,
        [
            _row(query_id="q1", system="rag", phase="phase_1", snapshot="snap-a", fingerprint="fp-a"),
            _row(query_id="q2", system="rag", phase="phase_1", snapshot="snap-b", fingerprint="fp-a"),
        ],
    )

    try:
        run_command("inspect-run", AppConfig(project_root=tmp_path), run_file=str(run_file))
    except ValueError as exc:
        assert "Inconsistent corpus_snapshot within system/phase cohort" in str(exc)
    else:
        raise AssertionError("Expected mixed snapshot within cohort to fail.")

    _write_run_jsonl(
        run_file,
        [
            _row(query_id="q1", system="rag", phase="phase_1", snapshot="snap-a", fingerprint="fp-a"),
            _row(query_id="q2", system="rag", phase="phase_1", snapshot="snap-a", fingerprint="fp-b"),
        ],
    )

    try:
        run_command("inspect-run", AppConfig(project_root=tmp_path), run_file=str(run_file))
    except ValueError as exc:
        assert "Inconsistent execution_fingerprint within system/phase cohort" in str(exc)
    else:
        raise AssertionError("Expected mixed execution fingerprint within cohort to fail.")
