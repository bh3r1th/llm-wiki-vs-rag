"""Pipeline entry points for RAG operations."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from llm_wiki_vs_rag.config import AppConfig
from llm_wiki_vs_rag.data.load_docs import corpus_order_token, fingerprint_document_batch, load_source_documents
from llm_wiki_vs_rag.llm.client import LLMClient
from llm_wiki_vs_rag.models import GenerationResult, QueryCase
from llm_wiki_vs_rag.paths import ProjectPaths
from llm_wiki_vs_rag.rag.indexing import build_in_memory_index, load_index, persist_index
from llm_wiki_vs_rag.rag.prompting import build_rag_prompt
from llm_wiki_vs_rag.rag.retrieve import retrieve_top_k


def _new_run_id(prefix: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return f"{prefix}-{timestamp}-{uuid4().hex[:8]}"


def _resolve_rag_snapshot_identity(paths: ProjectPaths) -> str:
    manifest_path = paths.artifacts_dir / "rag_index" / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    snapshot_id = str(payload.get("snapshot_id", "")).strip()
    if not snapshot_id:
        raise ValueError(f"Missing snapshot_id in canonical snapshot manifest for rag: {manifest_path}")
    return snapshot_id


def _write_query_artifacts(
    paths: ProjectPaths,
    run_id: str,
    query: QueryCase,
    prompt: str,
    answer: str,
    retrieved_chunks: list,
    requested_top_k: int,
    corpus_snapshot: str,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
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
        "mode": "rag",
        "question": query.question,
        "requested_top_k": requested_top_k,
        "returned_top_k": len(retrieved_chunks),
        "corpus_snapshot": corpus_snapshot,
        "token_usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return run_dir


def build_rag_index(config: AppConfig, paths: ProjectPaths):
    """Build and persist a local RAG index from source documents."""
    batch = load_source_documents(paths.raw_dir)
    snapshot_id = fingerprint_document_batch(batch)
    index = build_in_memory_index(
        batch=batch,
        chunk_size_chars=config.rag.chunk_size,
        chunk_overlap_chars=config.rag.chunk_overlap,
    )
    persist_index(
        index=index,
        artifacts_dir=paths.artifacts_dir,
        snapshot_id=snapshot_id,
        corpus_order=corpus_order_token(batch),
    )
    return index


def answer_rag_query(
    config: AppConfig,
    paths: ProjectPaths,
    query: QueryCase,
    corpus_snapshot: str | None = None,
) -> GenerationResult:
    """Answer a single query with the persisted RAG baseline."""
    index = load_index(paths.artifacts_dir)
    llm_client = LLMClient(config=config.llm)
    return _answer_rag_query_with_resources(
        config=config,
        paths=paths,
        query=query,
        index=index,
        llm_client=llm_client,
        corpus_snapshot=corpus_snapshot,
    )


def _answer_rag_query_with_resources(
    config: AppConfig,
    paths: ProjectPaths,
    query: QueryCase,
    index,
    llm_client: LLMClient,
    corpus_snapshot: str | None = None,
) -> GenerationResult:
    """Answer one query using preloaded RAG resources."""
    start = perf_counter()
    requested_top_k = config.retrieval_top_k()
    chunks = retrieve_top_k(index=index, query=query.question, top_k=requested_top_k)
    prompt = build_rag_prompt(question=query.question, chunks=chunks)
    llm_response = llm_client.generate_response(prompt, require_token_usage=True)
    answer = llm_response.text
    run_id = _new_run_id(prefix=query.query_id)
    run_dir = _write_query_artifacts(
        paths=paths,
        run_id=run_id,
        query=query,
        prompt=prompt,
        answer=answer,
        retrieved_chunks=chunks,
        requested_top_k=requested_top_k,
        corpus_snapshot=corpus_snapshot or _resolve_rag_snapshot_identity(paths),
        prompt_tokens=llm_response.token_usage.prompt_tokens,
        completion_tokens=llm_response.token_usage.completion_tokens,
        total_tokens=llm_response.token_usage.total_tokens,
    )
    latency_ms = (perf_counter() - start) * 1000.0

    return GenerationResult(
        query_id=query.query_id,
        answer=answer,
        mode="rag",
        used_context_ids=[chunk.chunk_id for chunk in chunks],
        run_id=run_id,
        latency_ms=round(latency_ms, 3),
        artifact_dir=str(run_dir),
        prompt_tokens=llm_response.token_usage.prompt_tokens,
        completion_tokens=llm_response.token_usage.completion_tokens,
        total_tokens=llm_response.token_usage.total_tokens,
    )


def run_rag_queries(
    config: AppConfig,
    paths: ProjectPaths,
    query_cases: list[QueryCase],
    corpus_snapshot: str | None = None,
) -> list[GenerationResult]:
    """Run the RAG baseline for a query set."""
    index = load_index(paths.artifacts_dir)
    llm_client = LLMClient(config=config.llm)
    return [
        _answer_rag_query_with_resources(
            config=config,
            paths=paths,
            query=query,
            index=index,
            llm_client=llm_client,
            corpus_snapshot=corpus_snapshot,
        )
        for query in query_cases
    ]
