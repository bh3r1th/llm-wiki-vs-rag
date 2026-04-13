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
        assert "cannot run with llm mock/stub mode" in str(exc)
    else:
        raise AssertionError("Expected benchmark command to fail fast when mock mode is enabled.")


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
