from __future__ import annotations

import json

from llm_wiki_vs_rag.config import AppConfig
from llm_wiki_vs_rag.eval.harness import load_query_cases, run_phase_1_rag_queries
from llm_wiki_vs_rag.eval.models import EvalQueryCase, RunOutputRecord
from llm_wiki_vs_rag.paths import ProjectPaths
from llm_wiki_vs_rag.runner import run_command


def _write_queries(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_query_loader_rejects_duplicate_query_phase_pair(tmp_path):
    query_file = tmp_path / "queries.jsonl"
    _write_queries(
        query_file,
        [
            {"query_id": "q1", "phase": "phase_1", "question": "Q1?", "category": "lookup"},
            {"query_id": "q1", "phase": "phase_1", "question": "Q1?", "category": "lookup"},
            {"query_id": "q1", "phase": "phase_2", "question": "Q1?", "category": "lookup"},
        ],
    )

    try:
        load_query_cases(query_file)
    except ValueError as exc:
        assert "unique per (query_id, phase)" in str(exc)
    else:
        raise AssertionError("Expected duplicate (query_id, phase) rows to fail.")


def test_query_loader_rejects_phase_content_mismatch(tmp_path):
    query_file = tmp_path / "queries.jsonl"
    _write_queries(
        query_file,
        [
            {"query_id": "q1", "phase": "phase_1", "question": "Q1?", "category": "lookup"},
            {"query_id": "q1", "phase": "phase_2", "question": "Q1 changed?", "category": "synthesis"},
        ],
    )

    try:
        load_query_cases(query_file)
    except ValueError as exc:
        assert "stable across phase_1 and phase_2" in str(exc)
    else:
        raise AssertionError("Expected cross-phase question/category mismatch to fail.")


def test_query_loader_rejects_missing_phase_counterpart(tmp_path):
    query_file = tmp_path / "queries.jsonl"
    _write_queries(
        query_file,
        [
            {"query_id": "q1", "phase": "phase_1", "question": "Q1?", "category": "lookup"},
            {"query_id": "q2", "phase": "phase_2", "question": "Q2?", "category": "synthesis"},
        ],
    )

    try:
        load_query_cases(query_file)
    except ValueError as exc:
        assert "same query_id cohort across phase_1 and phase_2" in str(exc)
    else:
        raise AssertionError("Expected missing cross-phase counterpart to fail.")


def test_query_loader_accepts_valid_same_queryset_across_phases(tmp_path):
    query_file = tmp_path / "queries.jsonl"
    _write_queries(
        query_file,
        [
            {"query_id": "q1", "phase": "phase_1", "question": "Q1?", "category": "lookup"},
            {"query_id": "q2", "phase": "phase_1", "question": "Q2?", "category": "contradiction"},
            {"query_id": "q1", "phase": "phase_2", "question": "Q1?", "category": "lookup"},
            {"query_id": "q2", "phase": "phase_2", "question": "Q2?", "category": "contradiction"},
        ],
    )

    loaded = load_query_cases(query_file)

    assert len(loaded) == 4


def test_validate_queries_command_runs_loader_contract_checks(tmp_path):
    query_file = tmp_path / "queries.jsonl"
    _write_queries(
        query_file,
        [
            {"query_id": "q1", "phase": "phase_1", "question": "Q1?", "category": "lookup"},
            {"query_id": "q1", "phase": "phase_2", "question": "Q1?", "category": "lookup"},
        ],
    )

    run_command("validate-queries", AppConfig(project_root=tmp_path), query_file=str(query_file))


def test_phase_specific_runner_accepts_single_phase_query_file(monkeypatch, tmp_path):
    single_phase_rows = [
        {"query_id": "q1", "phase": "phase_1", "question": "Q1?", "category": "lookup"},
        {"query_id": "q2", "phase": "phase_1", "question": "Q2?", "category": "synthesis"},
    ]
    query_cases = [EvalQueryCase.model_validate(row) for row in single_phase_rows]

    def _fake_run_queries_for_system(**_kwargs):
        return [
            RunOutputRecord(
                query_id="q1",
                system="rag",
                phase="phase_1",
                question="Q1?",
                category="lookup",
                answer="A1",
                metadata={
                    "execution_fingerprint": "sha256:exec",
                    "corpus_snapshot": "sha256:snap",
                    "corpus_order": "001",
                },
            )
        ]

    monkeypatch.setattr("llm_wiki_vs_rag.eval.harness.run_queries_for_system", _fake_run_queries_for_system)

    records = run_phase_1_rag_queries(
        config=AppConfig(project_root=tmp_path),
        paths=ProjectPaths(tmp_path),
        query_cases=query_cases,
    )

    assert len(records) == 1
