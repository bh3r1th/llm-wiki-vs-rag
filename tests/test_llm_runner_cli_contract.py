from __future__ import annotations

import json

from llm_wiki_vs_rag.cli.main import build_parser
from llm_wiki_vs_rag.config import AppConfig, LLMConfig
from llm_wiki_vs_rag.llm.client import LLMClient
from llm_wiki_vs_rag.runner import run_command


def test_deterministic_mock_mode_works_for_unit_tests():
    client = LLMClient(LLMConfig(provider="openai-compatible", mock_mode=True, mock_response='{"ok": true}'))

    assert client.generate("any prompt") == '{"ok": true}'
    assert client.generate_json("any prompt") == {"ok": True}


def test_generate_json_fails_on_non_json_mock_response():
    client = LLMClient(LLMConfig(provider="openai-compatible", mock_mode=True, mock_response="not json"))

    try:
        client.generate_json("prompt")
    except ValueError as exc:
        assert "not valid JSON" in str(exc)
    else:
        raise AssertionError("Expected strict JSON parse failure.")


def test_provider_response_token_usage_is_captured(monkeypatch):
    class _FakeHTTPResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(
                {
                    "choices": [{"message": {"content": "hello"}}],
                    "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
                }
            ).encode("utf-8")

    monkeypatch.setattr("llm_wiki_vs_rag.llm.client.request.urlopen", lambda *_args, **_kwargs: _FakeHTTPResponse())

    client = LLMClient(
        LLMConfig(provider="openai-compatible", base_url="https://example", api_key="k", model_name="m")
    )
    response = client.generate_response("prompt", require_token_usage=True)

    assert response.text == "hello"
    assert response.token_usage is not None
    assert response.token_usage.total_tokens == 18


def test_benchmark_generate_fails_when_provider_usage_is_missing(monkeypatch):
    class _FakeHTTPResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps({"choices": [{"message": {"content": "hello"}}]}).encode("utf-8")

    monkeypatch.setattr("llm_wiki_vs_rag.llm.client.request.urlopen", lambda *_args, **_kwargs: _FakeHTTPResponse())

    client = LLMClient(
        LLMConfig(provider="openai-compatible", base_url="https://example", api_key="k", model_name="m")
    )

    try:
        client.generate_response("prompt", require_token_usage=True)
    except ValueError as exc:
        assert "did not return token usage" in str(exc)
    else:
        raise AssertionError("Expected benchmark generation to fail when usage metadata is absent.")


def test_benchmark_commands_reject_mock_mode(tmp_path):
    config = AppConfig(project_root=tmp_path, llm=LLMConfig(provider="openai-compatible", mock_mode=True))
    query_file = tmp_path / "queries.jsonl"
    query_file.write_text(
        json.dumps({"query_id": "q1", "question": "Q?", "category": "policy", "phase": "phase_1"}) + "\n",
        encoding="utf-8",
    )

    try:
        run_command("run-rag-queries", config, query_file=str(query_file))
    except ValueError as exc:
        assert "cannot run with llm mock_mode enabled" in str(exc)
    else:
        raise AssertionError("Expected benchmark command to fail fast when mock mode is enabled.")


def test_benchmark_commands_validate_single_mock_control_mechanism(tmp_path):
    config = AppConfig(
        project_root=tmp_path,
        llm=LLMConfig(provider="mock", mock_mode=False, base_url="http://example", api_key="key", model_name="m"),
    )
    query_file = tmp_path / "queries.jsonl"
    query_file.write_text(
        json.dumps({"query_id": "q1", "question": "Q?", "category": "policy", "phase": "phase_1"}) + "\n",
        encoding="utf-8",
    )

    try:
        run_command("run-rag-queries", config, query_file=str(query_file))
    except ValueError as exc:
        assert "Unsupported benchmark LLM provider: mock" in str(exc)
    else:
        raise AssertionError("Expected benchmark validation to ignore provider alias and enforce mock_mode only.")


def test_benchmark_commands_require_provider_config(tmp_path):
    config = AppConfig(
        project_root=tmp_path,
        llm=LLMConfig(provider="openai-compatible", base_url=None, api_key=None),
    )
    query_file = tmp_path / "queries.jsonl"
    query_file.write_text(
        json.dumps({"query_id": "q1", "question": "Q?", "category": "policy", "phase": "phase_1"}) + "\n",
        encoding="utf-8",
    )

    try:
        run_command("run-wiki-queries", config, query_file=str(query_file))
    except ValueError as exc:
        assert "llm.base_url is required" in str(exc)
    else:
        raise AssertionError("Expected missing provider config to fail fast.")


def test_cli_output_args_are_optional_and_runner_defaults_are_used(monkeypatch, tmp_path):
    parser = build_parser()
    args = parser.parse_args(["run-rag-queries", "--query-file", str(tmp_path / "queries.jsonl")])
    assert args.output_file is None

    query_file = tmp_path / "queries.jsonl"
    query_file.write_text(
        json.dumps({"query_id": "q1", "question": "Q?", "category": "policy", "phase": "phase_1"}) + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr("llm_wiki_vs_rag.runner.run_queries_for_system", lambda **kwargs: [])

    captured: dict[str, object] = {}

    def _capture_save(records, output_path):
        captured["output_path"] = output_path

    monkeypatch.setattr("llm_wiki_vs_rag.runner.save_run_outputs", _capture_save)

    config = AppConfig(
        project_root=tmp_path,
        llm=LLMConfig(provider="openai-compatible", base_url="http://example", api_key="key"),
    )
    run_command("run-rag-queries", config, query_file=str(query_file))

    assert captured["output_path"] == tmp_path / "artifacts" / "run-rag-queries.jsonl"


def test_compare_systems_fails_on_mismatched_query_cohorts(tmp_path):
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
            }
        )
        + "\n",
        encoding="utf-8",
    )
    wiki_run_file = tmp_path / "wiki.jsonl"
    wiki_run_file.write_text(
        json.dumps(
            {
                "query_id": "q2",
                "system": "wiki",
                "phase": "phase_1",
                "question": "Q2",
                "category": "policy",
                "answer": "A2",
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

    config = AppConfig(project_root=tmp_path)
    try:
        run_command(
            "compare-systems",
            config,
            rag_run_file=str(rag_run_file),
            wiki_run_file=str(wiki_run_file),
            labels_file=str(labels_file),
        )
    except ValueError as exc:
        assert "mismatched (query_id, phase) cohorts" in str(exc)
    else:
        raise AssertionError("Expected compare-systems to fail when query cohorts differ.")


def test_compare_systems_fails_when_query_phase_pairs_do_not_match(tmp_path):
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
            }
        )
        + "\n"
        + json.dumps(
            {
                "query_id": "q2",
                "system": "rag",
                "phase": "phase_2",
                "question": "Q2",
                "category": "policy",
                "answer": "A2",
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
                "phase": "phase_2",
                "question": "Q1",
                "category": "policy",
                "answer": "A1",
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

    config = AppConfig(project_root=tmp_path)
    try:
        run_command(
            "compare-systems",
            config,
            rag_run_file=str(rag_run_file),
            wiki_run_file=str(wiki_run_file),
            labels_file=str(labels_file),
        )
    except ValueError as exc:
        assert "mismatched (query_id, phase) cohorts" in str(exc)
    else:
        raise AssertionError("Expected compare-systems to fail when query/phase pair cohorts differ.")


def test_compare_systems_fails_when_duplicate_system_rows_exist(tmp_path):
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
            }
        )
        + "\n"
        + json.dumps(
            {
                "query_id": "q1",
                "system": "rag",
                "phase": "phase_1",
                "question": "Q1 duplicate",
                "category": "policy",
                "answer": "A1 duplicate",
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
            }
        )
        + "\n",
        encoding="utf-8",
    )
    labels_file = tmp_path / "labels.csv"
    labels_file.write_text(
        "system,query_id,phase,accuracy,synthesis,latest_state,contradiction_detected,contradiction_resolved,compression_loss,provenance_fidelity,evaluator_notes\n"
        "rag,q1,phase_1,correct,full,correct,true,true,none,true,\n"
        "wiki,q1,phase_1,correct,full,correct,true,true,none,true,\n",
        encoding="utf-8",
    )

    config = AppConfig(project_root=tmp_path)
    try:
        run_command(
            "compare-systems",
            config,
            rag_run_file=str(rag_run_file),
            wiki_run_file=str(wiki_run_file),
            labels_file=str(labels_file),
        )
    except ValueError as exc:
        assert "duplicate (query_id, phase) rows" in str(exc)
        assert "q1" in str(exc)
    else:
        raise AssertionError("Expected compare-systems to fail on duplicate per-system rows.")


def test_compare_systems_fails_when_phase_snapshot_identity_missing(tmp_path):
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
                "metadata": {},
            }
        )
        + "\n"
        + json.dumps(
            {
                "query_id": "q2",
                "system": "rag",
                "phase": "phase_2",
                "question": "Q2",
                "category": "policy",
                "answer": "A2",
                "metadata": {},
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
                "metadata": {"corpus_snapshot": "wiki-a"},
            }
        )
        + "\n"
        + json.dumps(
            {
                "query_id": "q2",
                "system": "wiki",
                "phase": "phase_2",
                "question": "Q2",
                "category": "policy",
                "answer": "A2",
                "metadata": {"corpus_snapshot": "wiki-b"},
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
        assert "Missing corpus snapshot identity" in str(exc)
    else:
        raise AssertionError("Expected compare-systems to fail when snapshot identity is missing.")


def test_compare_systems_fails_when_phase_snapshot_mapping_inconsistent(tmp_path):
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
                "metadata": {"corpus_snapshot": "rag-phase-1"},
            }
        )
        + "\n"
        + json.dumps(
            {
                "query_id": "q2",
                "system": "rag",
                "phase": "phase_2",
                "question": "Q2",
                "category": "policy",
                "answer": "A2",
                "metadata": {"corpus_snapshot": "rag-phase-1"},
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
                "metadata": {"corpus_snapshot": "wiki-phase-1"},
            }
        )
        + "\n"
        + json.dumps(
            {
                "query_id": "q2",
                "system": "wiki",
                "phase": "phase_2",
                "question": "Q2",
                "category": "policy",
                "answer": "A2",
                "metadata": {"corpus_snapshot": "wiki-phase-2"},
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
        assert "distinct corpus snapshots for phase_1 and phase_2" in str(exc)
    else:
        raise AssertionError("Expected compare-systems to fail on inconsistent phase-to-snapshot mapping.")


def test_compare_systems_fails_when_query_text_or_category_mismatch_for_same_query_id_phase(tmp_path):
    rag_run_file = tmp_path / "rag.jsonl"
    rag_run_file.write_text(
        json.dumps(
            {
                "query_id": "q1",
                "system": "rag",
                "phase": "phase_1",
                "question": "What is policy A?",
                "category": "policy",
                "answer": "A1",
                "metadata": {"corpus_snapshot": "rag-phase-1"},
            }
        )
        + "\n"
        + json.dumps(
            {
                "query_id": "q2",
                "system": "rag",
                "phase": "phase_2",
                "question": "What is policy B?",
                "category": "policy",
                "answer": "A2",
                "metadata": {"corpus_snapshot": "rag-phase-2"},
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
                "question": "What changed for policy A?",
                "category": "history",
                "answer": "A1",
                "metadata": {"corpus_snapshot": "wiki-phase-1"},
            }
        )
        + "\n"
        + json.dumps(
            {
                "query_id": "q2",
                "system": "wiki",
                "phase": "phase_2",
                "question": "What is policy B?",
                "category": "policy",
                "answer": "A2",
                "metadata": {"corpus_snapshot": "wiki-phase-2"},
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
        assert "differ in question/category content" in str(exc)
        assert "q1" in str(exc)
    else:
        raise AssertionError("Expected compare-systems to fail when matched rows differ in question/category.")
