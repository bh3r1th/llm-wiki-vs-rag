"""Focused tests for RAG baseline behavior."""

import json
import os
import subprocess
import sys

import numpy as np

from llm_wiki_vs_rag.config import AppConfig
from llm_wiki_vs_rag.models import DocumentBatch, QueryCase, RetrievedChunk, SourceDocument
from llm_wiki_vs_rag.paths import ProjectPaths
from llm_wiki_vs_rag.rag.chunking import chunk_document
from llm_wiki_vs_rag.rag.indexing import RAGIndex, _embed_text, build_in_memory_index, persist_index
from llm_wiki_vs_rag.rag.pipeline import answer_rag_query, build_rag_index
from llm_wiki_vs_rag.rag.retrieve import retrieve_top_k


def test_chunking_is_deterministic(tmp_path):
    document = SourceDocument(doc_id="doc1", source_path=tmp_path / "doc1.txt", text="abcdefghijklmno")

    first = chunk_document(document=document, chunk_size_chars=6, chunk_overlap_chars=2)
    second = chunk_document(document=document, chunk_size_chars=6, chunk_overlap_chars=2)

    assert [chunk.chunk_id for chunk in first] == ["doc1:0", "doc1:1", "doc1:2", "doc1:3"]
    assert [chunk.text for chunk in first] == [chunk.text for chunk in second]
    assert first[0].position == {"chunk_index": 0, "start_char": 0, "end_char": 6}


def test_retrieve_top_k_returns_best_matches(monkeypatch):
    chunks = [
        RetrievedChunk(doc_id="d", chunk_id="d:0", text="apple banana"),
        RetrievedChunk(doc_id="d", chunk_id="d:1", text="banana banana"),
        RetrievedChunk(doc_id="d", chunk_id="d:2", text="orange grape"),
    ]

    index = RAGIndex(
        chunks=chunks,
        embeddings=np.array(
            [
                [0.8, 0.2],
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        backend="numpy",
    )

    monkeypatch.setattr("llm_wiki_vs_rag.rag.retrieve.embed_query", lambda _query: np.array([1.0, 0.0], dtype=np.float32))
    top = retrieve_top_k(index=index, query="banana", top_k=2)

    assert [chunk.chunk_id for chunk in top] == ["d:1", "d:0"]
    assert len(top) == 2


def test_embeddings_are_deterministic_across_python_hash_seeds():
    vector = _embed_text("Deterministic Token Test").tolist()
    script = (
        "import json;"
        "from llm_wiki_vs_rag.rag.indexing import _embed_text;"
        "print(json.dumps(_embed_text('Deterministic Token Test').tolist()))"
    )
    env_one = os.environ.copy()
    env_one["PYTHONHASHSEED"] = "1"
    env_two = os.environ.copy()
    env_two["PYTHONHASHSEED"] = "999"
    first = subprocess.check_output([sys.executable, "-c", script], text=True, env=env_one)
    second = subprocess.check_output([sys.executable, "-c", script], text=True, env=env_two)

    assert np.allclose(np.array(vector, dtype=np.float32), np.array(json.loads(first), dtype=np.float32))
    assert np.allclose(np.array(vector, dtype=np.float32), np.array(json.loads(second), dtype=np.float32))


def test_retrieve_top_k_enforces_explicit_cosine_similarity(monkeypatch):
    chunks = [
        RetrievedChunk(doc_id="d", chunk_id="d:0", text="c0"),
        RetrievedChunk(doc_id="d", chunk_id="d:1", text="c1"),
    ]
    index = RAGIndex(
        chunks=chunks,
        embeddings=np.array(
            [
                [5.0, 0.0],
                [4.0, 4.0],
            ],
            dtype=np.float32,
        ),
        backend="numpy",
    )
    monkeypatch.setattr("llm_wiki_vs_rag.rag.retrieve.embed_query", lambda _query: np.array([1.0, 0.0], dtype=np.float32))

    top = retrieve_top_k(index=index, query="q", top_k=2)
    assert [chunk.chunk_id for chunk in top] == ["d:0", "d:1"]
    assert top[0].score == 1.0
    assert top[1].score < 1.0


def test_build_in_memory_index_uses_numpy_backend_only(tmp_path):
    document = SourceDocument(doc_id="doc1", source_path=tmp_path / "doc1.txt", text="alpha beta gamma")
    index = build_in_memory_index(batch=DocumentBatch(documents=[document]), chunk_size_chars=100, chunk_overlap_chars=0)
    assert index.backend == "numpy"


def test_pipeline_saves_query_artifacts(tmp_path):
    paths = ProjectPaths(project_root=tmp_path)
    paths.ensure()

    raw_doc = paths.raw_dir / "001_doc.txt"
    raw_doc.write_text("alpha beta gamma delta", encoding="utf-8")

    config = AppConfig(project_root=tmp_path)
    batch = DocumentBatch(documents=[SourceDocument(doc_id="001_doc", source_path=raw_doc, text=raw_doc.read_text())])
    index = build_in_memory_index(batch=batch, chunk_size_chars=20, chunk_overlap_chars=5)
    persist_index(index=index, artifacts_dir=paths.artifacts_dir, snapshot_id="sha256:test")

    result = answer_rag_query(config=config, paths=paths, query=QueryCase(query_id="q1", question="alpha?"))
    assert result.query_id == "q1"

    run_dirs = list((paths.artifacts_dir / "rag_runs").iterdir())
    assert len(run_dirs) == 1

    run_dir = run_dirs[0]
    assert (run_dir / "retrieved_chunks.json").exists()
    assert (run_dir / "prompt.txt").exists()
    assert (run_dir / "answer.txt").exists()
    assert (run_dir / "metadata.json").exists()

    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["query_id"] == "q1"
    assert metadata["corpus_snapshot"] == "sha256:test"
    assert metadata["requested_top_k"] == config.rag.top_k
    assert metadata["returned_top_k"] <= metadata["requested_top_k"]


def test_build_rag_index_writes_canonical_snapshot_identity(tmp_path):
    paths = ProjectPaths(project_root=tmp_path)
    paths.ensure()
    (paths.raw_dir / "001_doc.txt").write_text("alpha", encoding="utf-8")
    (paths.raw_dir / "002_doc.txt").write_text("beta", encoding="utf-8")

    build_rag_index(config=AppConfig(project_root=tmp_path), paths=paths)

    manifest = json.loads((paths.artifacts_dir / "rag_index" / "manifest.json").read_text(encoding="utf-8"))
    assert str(manifest["snapshot_id"]).startswith("sha256:")
