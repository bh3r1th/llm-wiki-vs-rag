"""Focused tests for benchmark fairness and purity constraints."""

from __future__ import annotations

import json
import inspect
from pathlib import Path

from llm_wiki_vs_rag.config import AppConfig, LLMConfig, RAGConfig
from llm_wiki_vs_rag.data.load_docs import fingerprint_document_batch, load_source_documents
from llm_wiki_vs_rag.models import QueryCase, SourceDocument
from llm_wiki_vs_rag.paths import ProjectPaths
from llm_wiki_vs_rag.rag.pipeline import answer_rag_query, run_rag_queries
from llm_wiki_vs_rag.reproducibility import compute_execution_fingerprint
from llm_wiki_vs_rag.wiki.pipeline import ingest_wiki, run_wiki_queries
from llm_wiki_vs_rag.wiki.prompting import build_wiki_query_prompt
from llm_wiki_vs_rag.wiki.ingest import ingest_one_document


def test_locked_benchmark_uses_same_top_k_for_rag_and_wiki(monkeypatch, tmp_path):
    paths = ProjectPaths(project_root=tmp_path)
    paths.ensure()

    config = AppConfig(
        project_root=tmp_path,
        rag=RAGConfig(top_k=4),
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
    (paths.raw_dir / "001_doc.txt").write_text("alpha", encoding="utf-8")
    snapshot = fingerprint_document_batch(load_source_documents(paths.raw_dir))
    (paths.artifacts_dir / "rag_index").mkdir(parents=True, exist_ok=True)
    (paths.artifacts_dir / "rag_index" / "manifest.json").write_text(
        json.dumps(
            {
                "snapshot_id": snapshot,
                "execution_fingerprint": compute_execution_fingerprint(config=config, system="rag"),
            }
        ),
        encoding="utf-8",
    )
    (paths.wiki_dir / "snapshot.json").write_text(
        json.dumps(
            {
                "snapshot_id": snapshot,
                "execution_fingerprint": compute_execution_fingerprint(config=config, system="wiki"),
            }
        ),
        encoding="utf-8",
    )

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
    (paths.raw_dir / "001_doc.txt").write_text("alpha", encoding="utf-8")
    snapshot = fingerprint_document_batch(load_source_documents(paths.raw_dir))
    (paths.wiki_dir / "snapshot.json").write_text(
        json.dumps(
            {
                "snapshot_id": snapshot,
                "execution_fingerprint": compute_execution_fingerprint(config=config, system="wiki"),
            }
        ),
        encoding="utf-8",
    )

    results = run_wiki_queries(config=config, paths=paths, query_cases=[QueryCase(query_id="q1", question="Q?")])

    assert len(results) == 1
    assert results[0].mode == "wiki"
    assert results[0].answer == "wiki-only"
    metadata = (Path(results[0].artifact_dir) / "metadata.json").read_text(encoding="utf-8")
    assert f'"corpus_snapshot": "{snapshot}"' in metadata


def test_run_wiki_queries_signature_has_no_fallback_parameter():
    assert "use_rag_fallback" not in inspect.signature(run_wiki_queries).parameters


def test_run_wiki_queries_signature_has_no_snapshot_override_parameter():
    assert "corpus_snapshot" not in inspect.signature(run_wiki_queries).parameters


def test_answer_rag_query_signature_has_no_snapshot_override_parameter():
    assert "corpus_snapshot" not in inspect.signature(answer_rag_query).parameters


def test_run_rag_queries_signature_has_no_snapshot_override_parameter():
    assert "corpus_snapshot" not in inspect.signature(run_rag_queries).parameters


def test_retrieval_top_k_uses_shared_rag_budget():
    config = AppConfig(rag=RAGConfig(top_k=5))
    assert config.retrieval_top_k() == 5


def test_wiki_ingest_writes_canonical_snapshot_identity(monkeypatch, tmp_path):
    paths = ProjectPaths(project_root=tmp_path)
    paths.ensure()
    (paths.raw_dir / "001.txt").write_text("alpha", encoding="utf-8")
    (paths.raw_dir / "002.txt").write_text("beta", encoding="utf-8")
    monkeypatch.setattr("llm_wiki_vs_rag.wiki.pipeline.ingest_one_document", lambda **_kwargs: {"ok": True})

    ingest_wiki(config=AppConfig(project_root=tmp_path), paths=paths)

    payload = (paths.wiki_dir / "snapshot.json").read_text(encoding="utf-8")
    assert '"snapshot_id": "sha256:' in payload
    assert '"execution_fingerprint": "sha256:' in payload


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


def test_wiki_ingest_fails_on_empty_model_output_without_raw_copy_fallback(tmp_path):
    paths = ProjectPaths(project_root=tmp_path)
    paths.ensure()

    class _FakeLLM:
        def generate_json(self, _prompt):
            return {"pages_to_create": [], "pages_to_update": []}

    try:
        ingest_one_document(
            paths=paths,
            llm_client=_FakeLLM(),
            document=SourceDocument(doc_id="001", source_path=tmp_path / "001.txt", text="raw facts"),
            ingest_run_id="run-1",
            corpus_snapshot="sha256:snapshot",
        )
    except ValueError as exc:
        assert "Ingest model output is empty" in str(exc)
    else:
        raise AssertionError("Expected empty ingest output to fail.")

    failure_path = paths.artifacts_dir / "wiki_ingest" / "run-1" / "001" / "ingest_failure.json"
    assert failure_path.exists()
    assert not list(paths.wiki_dir.glob("*.md"))


def test_load_source_documents_fails_on_nonconforming_chronology_filenames(tmp_path):
    (tmp_path / "doc-alpha.txt").write_text("alpha", encoding="utf-8")
    (tmp_path / "002-event.txt").write_text("beta", encoding="utf-8")
    try:
        load_source_documents(tmp_path)
    except ValueError as exc:
        assert "chronology requires filename stems to start with a zero-padded numeric prefix" in str(exc)
    else:
        raise AssertionError("Expected chronology validation to fail on nonconforming filenames.")


def test_load_source_documents_retains_empty_raw_files_for_snapshot_accounting(tmp_path):
    (tmp_path / "001_event.txt").write_text("alpha", encoding="utf-8")
    (tmp_path / "002_event.txt").write_text("", encoding="utf-8")

    batch = load_source_documents(tmp_path)

    assert [doc.doc_id for doc in batch.documents] == ["001_event", "002_event"]
    assert batch.documents[1].text == ""
    assert [entry["filename"] for entry in batch.chronology] == ["001_event.txt", "002_event.txt"]


def test_empty_raw_files_change_snapshot_identity(tmp_path):
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    root_a.mkdir(parents=True, exist_ok=True)
    root_b.mkdir(parents=True, exist_ok=True)
    (root_a / "001_event.txt").write_text("alpha", encoding="utf-8")
    (root_b / "001_event.txt").write_text("alpha", encoding="utf-8")
    (root_b / "002_event.txt").write_text("", encoding="utf-8")

    snapshot_a = fingerprint_document_batch(load_source_documents(root_a))
    snapshot_b = fingerprint_document_batch(load_source_documents(root_b))

    assert snapshot_a != snapshot_b


def test_execution_fingerprint_changes_when_chunking_changes_but_snapshot_stays_same(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "001_doc.txt").write_text("alpha", encoding="utf-8")
    snapshot_a = fingerprint_document_batch(load_source_documents(raw_dir))
    snapshot_b = fingerprint_document_batch(load_source_documents(raw_dir))
    config_a = AppConfig(project_root=tmp_path, rag=RAGConfig(chunk_size=100, chunk_overlap=10))
    config_b = AppConfig(project_root=tmp_path, rag=RAGConfig(chunk_size=120, chunk_overlap=10))

    assert snapshot_a == snapshot_b
    assert compute_execution_fingerprint(config=config_a, system="rag") != compute_execution_fingerprint(
        config=config_b, system="rag"
    )


def test_execution_fingerprint_uses_config_model_when_present(monkeypatch, tmp_path):
    monkeypatch.delenv("LLM_MODEL", raising=False)
    config = AppConfig(project_root=tmp_path, llm=LLMConfig(provider="openai-compatible", model_name="cfg-model"))

    fingerprint_a = compute_execution_fingerprint(config=config, system="rag")
    fingerprint_b = compute_execution_fingerprint(config=config, system="rag")

    assert fingerprint_a == fingerprint_b


def test_execution_fingerprint_uses_env_model_when_config_model_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_MODEL", "env-model")
    llm_config = LLMConfig.model_construct(
        provider="openai-compatible",
        model_name=None,
        temperature=0.0,
        timeout_seconds=30,
        base_url=None,
        api_key=None,
        mock_mode=False,
        mock_response='{"pages_to_create": [], "pages_to_update": [], "index_note": "", "log_note": ""}',
    )
    config = AppConfig(project_root=tmp_path, llm=llm_config)

    env_fingerprint = compute_execution_fingerprint(config=config, system="rag")
    cfg_fingerprint = compute_execution_fingerprint(
        config=AppConfig(project_root=tmp_path, llm=LLMConfig(provider="openai-compatible", model_name="cfg-model")),
        system="rag",
    )

    assert env_fingerprint != cfg_fingerprint


def test_execution_fingerprint_prefers_config_model_over_env_model(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_MODEL", "env-model")
    config = AppConfig(project_root=tmp_path, llm=LLMConfig(provider="openai-compatible", model_name="cfg-model"))

    fingerprint_with_env_set = compute_execution_fingerprint(config=config, system="rag")
    fingerprint_without_env = compute_execution_fingerprint(
        config=AppConfig(project_root=tmp_path, llm=LLMConfig(provider="openai-compatible", model_name="cfg-model")),
        system="rag",
    )

    assert fingerprint_with_env_set == fingerprint_without_env


def test_rag_query_fails_if_raw_corpus_drifted_since_index_build(monkeypatch, tmp_path):
    paths = ProjectPaths(project_root=tmp_path)
    paths.ensure()
    config = AppConfig(project_root=tmp_path, llm=LLMConfig(mock_mode=True, mock_response="ok"))
    (paths.raw_dir / "001_doc.txt").write_text("alpha", encoding="utf-8")
    snapshot = fingerprint_document_batch(load_source_documents(paths.raw_dir))
    (paths.artifacts_dir / "rag_index").mkdir(parents=True, exist_ok=True)
    (paths.artifacts_dir / "rag_index" / "manifest.json").write_text(
        json.dumps(
            {
                "snapshot_id": snapshot,
                "execution_fingerprint": compute_execution_fingerprint(config=config, system="rag"),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("llm_wiki_vs_rag.rag.pipeline.load_index", lambda _artifacts_dir: object())
    monkeypatch.setattr("llm_wiki_vs_rag.rag.pipeline.retrieve_top_k", lambda **_kwargs: [])
    (paths.raw_dir / "001_doc.txt").write_text("beta", encoding="utf-8")

    try:
        run_rag_queries(config=config, paths=paths, query_cases=[QueryCase(query_id="q1", question="Q?")])
    except ValueError as exc:
        assert "Raw corpus snapshot drift detected" in str(exc)
    else:
        raise AssertionError("Expected RAG queries to fail when raw corpus drift is detected.")


def test_wiki_query_fails_if_raw_corpus_drifted_since_ingest(monkeypatch, tmp_path):
    paths = ProjectPaths(project_root=tmp_path)
    paths.ensure()
    config = AppConfig(project_root=tmp_path, llm=LLMConfig(mock_mode=True, mock_response="ok"))
    (paths.raw_dir / "001_doc.txt").write_text("alpha", encoding="utf-8")
    snapshot = fingerprint_document_batch(load_source_documents(paths.raw_dir))
    (paths.wiki_dir / "snapshot.json").write_text(
        json.dumps(
            {
                "snapshot_id": snapshot,
                "execution_fingerprint": compute_execution_fingerprint(config=config, system="wiki"),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("llm_wiki_vs_rag.wiki.pipeline.load_pages", lambda _wiki_dir: [])
    monkeypatch.setattr("llm_wiki_vs_rag.wiki.pipeline.retrieve_wiki_pages", lambda pages, query, top_k: [])
    (paths.raw_dir / "001_doc.txt").write_text("beta", encoding="utf-8")

    try:
        run_wiki_queries(config=config, paths=paths, query_cases=[QueryCase(query_id="q1", question="Q?")])
    except ValueError as exc:
        assert "Raw corpus snapshot drift detected" in str(exc)
    else:
        raise AssertionError("Expected Wiki queries to fail when raw corpus drift is detected.")


def test_rag_query_fails_if_runtime_execution_fingerprint_differs_from_manifest(monkeypatch, tmp_path):
    paths = ProjectPaths(project_root=tmp_path)
    paths.ensure()
    config = AppConfig(project_root=tmp_path, llm=LLMConfig(mock_mode=True, mock_response="ok"))
    (paths.raw_dir / "001_doc.txt").write_text("alpha", encoding="utf-8")
    snapshot = fingerprint_document_batch(load_source_documents(paths.raw_dir))
    (paths.artifacts_dir / "rag_index").mkdir(parents=True, exist_ok=True)
    (paths.artifacts_dir / "rag_index" / "manifest.json").write_text(
        json.dumps({"snapshot_id": snapshot, "execution_fingerprint": "sha256:stale-fingerprint"}),
        encoding="utf-8",
    )
    monkeypatch.setattr("llm_wiki_vs_rag.rag.pipeline.load_index", lambda _artifacts_dir: object())
    monkeypatch.setattr("llm_wiki_vs_rag.rag.pipeline.retrieve_top_k", lambda **_kwargs: [])

    try:
        run_rag_queries(config=config, paths=paths, query_cases=[QueryCase(query_id="q1", question="Q?")])
    except ValueError as exc:
        assert "Execution fingerprint mismatch at query runtime" in str(exc)
        assert "system=rag" in str(exc)
        assert "manifest_fingerprint=sha256:stale-fingerprint" in str(exc)
    else:
        raise AssertionError("Expected RAG queries to fail when runtime execution fingerprint mismatches manifest.")


def test_wiki_query_fails_if_runtime_execution_fingerprint_differs_from_manifest(monkeypatch, tmp_path):
    paths = ProjectPaths(project_root=tmp_path)
    paths.ensure()
    config = AppConfig(project_root=tmp_path, llm=LLMConfig(mock_mode=True, mock_response="ok"))
    (paths.raw_dir / "001_doc.txt").write_text("alpha", encoding="utf-8")
    snapshot = fingerprint_document_batch(load_source_documents(paths.raw_dir))
    (paths.wiki_dir / "snapshot.json").write_text(
        json.dumps({"snapshot_id": snapshot, "execution_fingerprint": "sha256:stale-fingerprint"}),
        encoding="utf-8",
    )
    monkeypatch.setattr("llm_wiki_vs_rag.wiki.pipeline.load_pages", lambda _wiki_dir: [])
    monkeypatch.setattr("llm_wiki_vs_rag.wiki.pipeline.retrieve_wiki_pages", lambda pages, query, top_k: [])

    try:
        run_wiki_queries(config=config, paths=paths, query_cases=[QueryCase(query_id="q1", question="Q?")])
    except ValueError as exc:
        assert "Execution fingerprint mismatch at query runtime" in str(exc)
        assert "system=wiki" in str(exc)
        assert "manifest_fingerprint=sha256:stale-fingerprint" in str(exc)
    else:
        raise AssertionError("Expected Wiki queries to fail when runtime execution fingerprint mismatches manifest.")


def test_runtime_execution_fingerprint_mismatch_detects_model_top_k_and_chunking_drift(monkeypatch, tmp_path):
    paths = ProjectPaths(project_root=tmp_path)
    paths.ensure()
    (paths.raw_dir / "001_doc.txt").write_text("alpha", encoding="utf-8")
    snapshot = fingerprint_document_batch(load_source_documents(paths.raw_dir))
    baseline = AppConfig(
        project_root=tmp_path,
        llm=LLMConfig(provider="openai-compatible", model_name="baseline-model", mock_mode=True, mock_response="ok"),
        rag=RAGConfig(top_k=5, chunk_size=500, chunk_overlap=50),
    )
    drifted = AppConfig(
        project_root=tmp_path,
        llm=LLMConfig(provider="openai-compatible", model_name="new-model", mock_mode=True, mock_response="ok"),
        rag=RAGConfig(top_k=7, chunk_size=700, chunk_overlap=10),
    )
    (paths.artifacts_dir / "rag_index").mkdir(parents=True, exist_ok=True)
    (paths.artifacts_dir / "rag_index" / "manifest.json").write_text(
        json.dumps(
            {
                "snapshot_id": snapshot,
                "execution_fingerprint": compute_execution_fingerprint(config=baseline, system="rag"),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("llm_wiki_vs_rag.rag.pipeline.load_index", lambda _artifacts_dir: object())
    monkeypatch.setattr("llm_wiki_vs_rag.rag.pipeline.retrieve_top_k", lambda **_kwargs: [])

    try:
        run_rag_queries(config=drifted, paths=paths, query_cases=[QueryCase(query_id="q1", question="Q?")])
    except ValueError as exc:
        assert "Execution fingerprint mismatch at query runtime" in str(exc)
    else:
        raise AssertionError("Expected runtime fingerprint drift to fail for model/top_k/chunking changes.")
