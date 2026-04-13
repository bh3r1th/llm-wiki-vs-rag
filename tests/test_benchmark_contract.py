"""Focused tests for benchmark fairness and purity constraints."""

from __future__ import annotations

from llm_wiki_vs_rag.config import AppConfig, BenchmarkConfig, RAGConfig, WikiConfig
from llm_wiki_vs_rag.eval.harness import run_queries_for_system
from llm_wiki_vs_rag.eval.models import EvalQueryCase
from llm_wiki_vs_rag.models import QueryCase
from llm_wiki_vs_rag.paths import ProjectPaths
from llm_wiki_vs_rag.rag.pipeline import answer_rag_query
from llm_wiki_vs_rag.wiki.pipeline import run_wiki_queries
from llm_wiki_vs_rag.wiki.prompting import build_wiki_query_prompt


def test_locked_benchmark_uses_same_top_k_for_rag_and_wiki(monkeypatch, tmp_path):
    paths = ProjectPaths(project_root=tmp_path)
    paths.ensure()
    (paths.artifacts_dir / "rag_index.json").write_text('{"chunks": [], "backend": "numpy"}', encoding="utf-8")

    config = AppConfig(
        project_root=tmp_path,
        rag=RAGConfig(top_k=4),
        wiki=WikiConfig(query_top_k=4),
    )
    observed: list[int] = []

    monkeypatch.setattr("llm_wiki_vs_rag.rag.pipeline.load_index", lambda _artifacts_dir: object())
    monkeypatch.setattr(
        "llm_wiki_vs_rag.rag.pipeline.retrieve_top_k",
        lambda index, query, top_k: observed.append(top_k) or [],
    )
    monkeypatch.setattr("llm_wiki_vs_rag.wiki.pipeline.load_pages", lambda _wiki_dir: [])
    monkeypatch.setattr(
        "llm_wiki_vs_rag.wiki.pipeline.retrieve_wiki_pages",
        lambda pages, query, top_k: observed.append(top_k) or [],
    )

    answer_rag_query(config=config, paths=paths, query=QueryCase(query_id="q-rag", question="Q?"))
    run_wiki_queries(config=config, paths=paths, query_cases=[QueryCase(query_id="q-wiki", question="Q?")])

    assert observed == [4, 4]


def test_wiki_query_prompt_has_refusal_and_latest_state_guards():
    prompt = build_wiki_query_prompt(question="What changed?", pages=[])
    assert "Answer only from the wiki context below." in prompt
    assert "reply exactly: INSUFFICIENT_EVIDENCE." in prompt
    assert "Do not invent latest-state, current-status, or time-sensitive claims" in prompt


def test_locked_benchmark_wiki_never_uses_rag_fallback(monkeypatch, tmp_path):
    paths = ProjectPaths(project_root=tmp_path)
    paths.ensure()
    config = AppConfig(
        project_root=tmp_path,
        benchmark=BenchmarkConfig(locked=True),
        wiki=WikiConfig(allow_rag_fallback=True),
    )
    monkeypatch.setattr("llm_wiki_vs_rag.wiki.pipeline.load_pages", lambda _wiki_dir: [])
    monkeypatch.setattr("llm_wiki_vs_rag.wiki.pipeline.retrieve_wiki_pages", lambda pages, query, top_k: [])
    monkeypatch.setattr("llm_wiki_vs_rag.wiki.pipeline.answer_rag_query", lambda **kwargs: None)

    query_cases = [EvalQueryCase(query_id="q1", question="Q?", category="policy", phase="phase_1")]
    try:
        run_queries_for_system(config=config, paths=paths, query_cases=query_cases, system="wiki")
    except ValueError as exc:
        assert "forbids wiki->RAG fallback" in str(exc)
    else:
        raise AssertionError("Expected locked benchmark mode to reject wiki fallback.")


def test_locked_benchmark_fails_fast_on_top_k_parity_break():
    config = AppConfig(
        rag=RAGConfig(top_k=5),
        wiki=WikiConfig(query_top_k=3),
        benchmark=BenchmarkConfig(locked=True),
    )
    try:
        config.retrieval_top_k()
    except ValueError as exc:
        assert "retrieval parity" in str(exc)
    else:
        raise AssertionError("Expected parity mismatch to fail fast in locked benchmark mode.")
