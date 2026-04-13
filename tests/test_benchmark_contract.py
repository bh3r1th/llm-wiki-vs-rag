"""Focused tests for benchmark fairness and purity constraints."""

from __future__ import annotations

import json
import inspect
from pathlib import Path

from llm_wiki_vs_rag.config import AppConfig, BenchmarkConfig, LLMConfig, RAGConfig, WikiConfig
from llm_wiki_vs_rag.data.load_docs import fingerprint_document_batch, load_source_documents
from llm_wiki_vs_rag.models import QueryCase
from llm_wiki_vs_rag.paths import ProjectPaths
from llm_wiki_vs_rag.rag.pipeline import answer_rag_query
from llm_wiki_vs_rag.wiki.pipeline import ingest_wiki, run_wiki_queries
from llm_wiki_vs_rag.wiki.prompting import build_wiki_query_prompt


def test_locked_benchmark_uses_same_top_k_for_rag_and_wiki(monkeypatch, tmp_path):
    paths = ProjectPaths(project_root=tmp_path)
    paths.ensure()

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
    (paths.artifacts_dir / "rag_index").mkdir(parents=True, exist_ok=True)
    (paths.artifacts_dir / "rag_index" / "manifest.json").write_text(
        '{"snapshot_id": "sha256:rag-test"}',
        encoding="utf-8",
    )
    (paths.wiki_dir / "snapshot.json").write_text('{"snapshot_id": "sha256:wiki-test"}', encoding="utf-8")

    answer_rag_query(config=config, paths=paths, query=QueryCase(query_id="q-rag", question="Q?"))
    run_wiki_queries(config=config, paths=paths, query_cases=[QueryCase(query_id="q-wiki", question="Q?")])

    assert observed == [4, 4]


def test_wiki_query_prompt_has_refusal_and_latest_state_guards():
    prompt = build_wiki_query_prompt(question="What changed?", pages=[])
    assert "Answer only from the wiki context below." in prompt
    assert "reply exactly: INSUFFICIENT_EVIDENCE." in prompt
    assert "Do not invent latest-state, current-status, or time-sensitive claims" in prompt


def test_benchmark_wiki_query_path_has_no_fallback_behavior(monkeypatch, tmp_path):
    paths = ProjectPaths(project_root=tmp_path)
    paths.ensure()
    config = AppConfig(project_root=tmp_path, llm=LLMConfig(mock_mode=True, mock_response="wiki-only"))
    monkeypatch.setattr("llm_wiki_vs_rag.wiki.pipeline.load_pages", lambda _wiki_dir: [])
    monkeypatch.setattr("llm_wiki_vs_rag.wiki.pipeline.retrieve_wiki_pages", lambda pages, query, top_k: [])
    (paths.wiki_dir / "snapshot.json").write_text('{"snapshot_id": "sha256:wiki-test"}', encoding="utf-8")

    results = run_wiki_queries(config=config, paths=paths, query_cases=[QueryCase(query_id="q1", question="Q?")])

    assert len(results) == 1
    assert results[0].mode == "wiki"
    assert results[0].answer == "wiki-only"
    metadata = (Path(results[0].artifact_dir) / "metadata.json").read_text(encoding="utf-8")
    assert '"corpus_snapshot": "sha256:wiki-test"' in metadata


def test_run_wiki_queries_signature_has_no_fallback_parameter():
    assert "use_rag_fallback" not in inspect.signature(run_wiki_queries).parameters


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


def test_wiki_ingest_writes_canonical_snapshot_identity(monkeypatch, tmp_path):
    paths = ProjectPaths(project_root=tmp_path)
    paths.ensure()
    (paths.raw_dir / "001.txt").write_text("alpha", encoding="utf-8")
    (paths.raw_dir / "002.txt").write_text("beta", encoding="utf-8")
    monkeypatch.setattr("llm_wiki_vs_rag.wiki.pipeline.ingest_one_document", lambda **_kwargs: {"ok": True})

    ingest_wiki(config=AppConfig(project_root=tmp_path), paths=paths)

    payload = (paths.wiki_dir / "snapshot.json").read_text(encoding="utf-8")
    assert '"snapshot_id": "sha256:' in payload


def test_snapshot_fingerprint_is_content_based_across_different_roots(tmp_path):
    root_a = tmp_path / "clone_a"
    root_b = tmp_path / "clone_b"
    root_a.mkdir(parents=True, exist_ok=True)
    root_b.mkdir(parents=True, exist_ok=True)
    (root_a / "001.txt").write_text("alpha", encoding="utf-8")
    (root_a / "002.md").write_text("beta", encoding="utf-8")
    (root_b / "001.txt").write_text("alpha", encoding="utf-8")
    (root_b / "002.md").write_text("beta", encoding="utf-8")

    batch_a = load_source_documents(root_a)
    batch_b = load_source_documents(root_b)

    assert fingerprint_document_batch(batch_a) == fingerprint_document_batch(batch_b)


def test_wiki_ingest_artifacts_preserve_rerun_history_for_same_doc_id(monkeypatch, tmp_path):
    paths = ProjectPaths(project_root=tmp_path)
    paths.ensure()
    (paths.raw_dir / "001.txt").write_text("alpha", encoding="utf-8")

    def _fake_ingest_one_document(*, paths, llm_client, document, ingest_run_id, corpus_snapshot):
        artifact_dir = paths.artifacts_dir / "wiki_ingest" / ingest_run_id / document.doc_id
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "applied_changes.json").write_text(
            json.dumps({"doc_id": document.doc_id, "ingest_run_id": ingest_run_id, "corpus_snapshot": corpus_snapshot}),
            encoding="utf-8",
        )
        return {"doc_id": document.doc_id}

    monkeypatch.setattr("llm_wiki_vs_rag.wiki.pipeline.ingest_one_document", _fake_ingest_one_document)

    ingest_wiki(config=AppConfig(project_root=tmp_path), paths=paths)
    ingest_wiki(config=AppConfig(project_root=tmp_path), paths=paths)

    doc_artifacts = list((paths.artifacts_dir / "wiki_ingest").glob("*/001/applied_changes.json"))
    assert len(doc_artifacts) == 2
    run_ids = {
        json.loads(artifact.read_text(encoding="utf-8"))["ingest_run_id"]
        for artifact in doc_artifacts
    }
    assert len(run_ids) == 2
