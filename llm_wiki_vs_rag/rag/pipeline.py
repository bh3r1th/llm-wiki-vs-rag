"""Pipeline entry points for RAG operations."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from llm_wiki_vs_rag.config import AppConfig
from llm_wiki_vs_rag.data.load_docs import load_source_documents
from llm_wiki_vs_rag.llm.client import LLMClient
from llm_wiki_vs_rag.models import GenerationResult, QueryCase
from llm_wiki_vs_rag.paths import ProjectPaths
from llm_wiki_vs_rag.rag.indexing import build_in_memory_index, load_index, persist_index
from llm_wiki_vs_rag.rag.prompting import build_rag_prompt
from llm_wiki_vs_rag.rag.retrieve import retrieve_top_k


def _new_run_id(prefix: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return f"{prefix}-{timestamp}-{uuid4().hex[:8]}"


def _write_query_artifacts(
    paths: ProjectPaths,
    run_id: str,
    query: QueryCase,
    prompt: str,
    answer: str,
    retrieved_chunks: list,
) -> Path:
    run_dir = paths.artifacts_dir / "rag_runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "retrieved_chunks.json").write_text(
        json.dumps([chunk.model_dump(mode="json") for chunk in retrieved_chunks], indent=2),
        encoding="utf-8",
    )
    (run_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    (run_dir / "answer.txt").write_text(answer, encoding="utf-8")
    metadata = {
        "run_id": run_id,
        "query_id": query.query_id,
        "question": query.question,
        "top_k": len(retrieved_chunks),
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return run_dir


def build_rag_index(config: AppConfig, paths: ProjectPaths):
    """Build and persist a local RAG index from source documents."""
    batch = load_source_documents(paths.raw_dir)
    index = build_in_memory_index(
        batch=batch,
        chunk_size_chars=config.rag.chunk_size,
        chunk_overlap_chars=config.rag.chunk_overlap,
    )
    persist_index(index=index, artifacts_dir=paths.artifacts_dir)
    return index


def answer_rag_query(config: AppConfig, paths: ProjectPaths, query: QueryCase) -> GenerationResult:
    """Answer a single query with the persisted RAG baseline."""
    start = perf_counter()
    index = load_index(paths.artifacts_dir)
    llm_client = LLMClient(config=config.llm)
    chunks = retrieve_top_k(index=index, query=query.question, top_k=config.retrieval_top_k())
    prompt = build_rag_prompt(question=query.question, chunks=chunks)
    answer = llm_client.generate(prompt)
    run_id = _new_run_id(prefix=query.query_id)
    run_dir = _write_query_artifacts(paths=paths, run_id=run_id, query=query, prompt=prompt, answer=answer, retrieved_chunks=chunks)
    latency_ms = (perf_counter() - start) * 1000.0

    return GenerationResult(
        query_id=query.query_id,
        answer=answer,
        mode="rag",
        used_context_ids=[chunk.chunk_id for chunk in chunks],
        run_id=run_id,
        latency_ms=round(latency_ms, 3),
        artifact_dir=str(run_dir),
    )


def run_rag_queries(config: AppConfig, paths: ProjectPaths, query_cases: list[QueryCase]) -> list[GenerationResult]:
    """Run the RAG baseline for a query set."""
    return [answer_rag_query(config=config, paths=paths, query=query) for query in query_cases]
