from __future__ import annotations

import json

from llm_wiki_vs_rag.config import AppConfig, LLMConfig
from llm_wiki_vs_rag.eval.harness import load_run_outputs
from llm_wiki_vs_rag.eval.models import RunOutputRecord
from llm_wiki_vs_rag.runner import run_command


def _write_queries(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _full_query_rows():
    rows = []
    categories = ["lookup", "synthesis", "latest_state", "contradiction"]
    for category in categories:
        for idx in range(1, 4):
            query_id = f"{category}-q{idx}"
            rows.append(
                {
                    "query_id": query_id,
                    "phase": "phase_1",
                    "question": f"{category} question {idx}?",
                    "category": category,
                }
            )
            rows.append(
                {
                    "query_id": query_id,
                    "phase": "phase_2",
                    "question": f"{category} question {idx}?",
                    "category": category,
                }
            )
    return rows


def test_smoke_subset_keeps_both_phases_for_selected_query_ids(tmp_path):
    query_file = tmp_path / "queries.jsonl"
    output_file = tmp_path / "smoke.jsonl"
    _write_queries(query_file, _full_query_rows())

    run_command("smoke-queries", AppConfig(project_root=tmp_path), query_file=str(query_file), output_file=str(output_file))

    lines = [json.loads(line) for line in output_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_query = {}
    for row in lines:
        by_query.setdefault(row["query_id"], set()).add(row["phase"])

    assert len(by_query) == 8
    assert all(phases == {"phase_1", "phase_2"} for phases in by_query.values())


def test_smoke_subset_category_counts_are_correct(tmp_path):
    query_file = tmp_path / "queries.jsonl"
    output_file = tmp_path / "smoke.jsonl"
    _write_queries(query_file, _full_query_rows())

    run_command("smoke-queries", AppConfig(project_root=tmp_path), query_file=str(query_file), output_file=str(output_file))

    lines = [json.loads(line) for line in output_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    phase_1_rows = [row for row in lines if row["phase"] == "phase_1"]
    counts = {}
    for row in phase_1_rows:
        counts[row["category"]] = counts.get(row["category"], 0) + 1

    assert counts == {"lookup": 2, "synthesis": 2, "latest_state": 2, "contradiction": 2}


def test_benchmark_phase_run_rejects_mixed_phase_input(tmp_path):
    query_file = tmp_path / "queries.jsonl"
    _write_queries(
        query_file,
        [
            {"query_id": "q1", "phase": "phase_1", "question": "Q1?", "category": "lookup"},
            {"query_id": "q1", "phase": "phase_2", "question": "Q1?", "category": "lookup"},
        ],
    )

    config = AppConfig(
        project_root=tmp_path,
        llm=LLMConfig(provider="openai-compatible", base_url="http://example", api_key="k"),
    )

    try:
        run_command(
            "benchmark-phase-run",
            config,
            system="rag",
            phase="phase_1",
            query_file=str(query_file),
        )
    except ValueError as exc:
        assert "Phase-targeted benchmark execution requires all query rows to match the requested phase" in str(exc)
    else:
        raise AssertionError("Expected benchmark-phase-run to reject mixed-phase query rows.")


def test_benchmark_phase_run_requires_explicit_phase(tmp_path):
    query_file = tmp_path / "queries.jsonl"
    _write_queries(
        query_file,
        [
            {"query_id": "q1", "phase": "phase_1", "question": "Q1?", "category": "lookup"},
            {"query_id": "q1", "phase": "phase_2", "question": "Q1?", "category": "lookup"},
        ],
    )

    config = AppConfig(
        project_root=tmp_path,
        llm=LLMConfig(provider="openai-compatible", base_url="http://example", api_key="k"),
    )

    try:
        run_command(
            "benchmark-phase-run",
            config,
            system="rag",
            query_file=str(query_file),
        )
    except ValueError as exc:
        assert "requires explicit --phase" in str(exc)
    else:
        raise AssertionError("Expected benchmark-phase-run to require explicit phase.")


def test_valid_phase_specific_run_succeeds(monkeypatch, tmp_path):
    query_file = tmp_path / "queries_phase_1.jsonl"
    output_file = tmp_path / "run.jsonl"
    _write_queries(
        query_file,
        [
            {"query_id": "q1", "phase": "phase_1", "question": "Q1?", "category": "lookup"},
            {"query_id": "q2", "phase": "phase_1", "question": "Q2?", "category": "synthesis"},
        ],
    )

    def _fake_run(**_kwargs):
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

    monkeypatch.setattr("llm_wiki_vs_rag.runner.run_phase_1_rag_queries", _fake_run)

    config = AppConfig(
        project_root=tmp_path,
        llm=LLMConfig(provider="openai-compatible", base_url="http://example", api_key="k"),
    )

    run_command(
        "benchmark-phase-run",
        config,
        system="rag",
        phase="phase_1",
        query_file=str(query_file),
        output_file=str(output_file),
    )

    loaded = load_run_outputs(output_file)
    assert len(loaded) == 1
    assert loaded[0].phase == "phase_1"
    assert loaded[0].system == "rag"
