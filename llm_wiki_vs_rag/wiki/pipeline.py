"""Pipeline entry points for ingest-time wiki synthesis and querying."""

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
from llm_wiki_vs_rag.wiki.ingest import ingest_one_document
from llm_wiki_vs_rag.wiki.pages import load_pages
from llm_wiki_vs_rag.wiki.prompting import build_wiki_query_prompt
from llm_wiki_vs_rag.wiki.retrieve import retrieve_wiki_pages


def _new_run_id(query_id: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return f"{query_id}-{timestamp}-{uuid4().hex[:8]}"


def _new_ingest_run_id(snapshot_id: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    snapshot_slug = snapshot_id.replace(":", "_")
    return f"{snapshot_slug}-{timestamp}-{uuid4().hex[:8]}"


def _resolve_wiki_snapshot_identity(paths: ProjectPaths) -> str:
    manifest_path = paths.wiki_dir / "snapshot.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    snapshot_id = str(payload.get("snapshot_id", "")).strip()
    if not snapshot_id:
        raise ValueError(f"Missing snapshot_id in canonical snapshot manifest for wiki: {manifest_path}")
    return snapshot_id


def _write_wiki_snapshot_manifest(wiki_dir: Path, snapshot_id: str, corpus_order: str | None = None) -> None:
    manifest_path = wiki_dir / "snapshot.json"
    manifest_path.write_text(
        json.dumps({"snapshot_id": snapshot_id, "corpus_order": corpus_order}, indent=2),
        encoding="utf-8",
    )


def ingest_wiki(config: AppConfig, paths: ProjectPaths):
    """Process raw docs sequentially and update markdown wiki incrementally."""
    batch = load_source_documents(paths.raw_dir)
    snapshot_id = fingerprint_document_batch(batch)
    ingest_run_id = _new_ingest_run_id(snapshot_id)
    llm_client = LLMClient(config=config.llm)
    summaries = []
    for document in batch.documents:
        summaries.append(
            ingest_one_document(
                paths=paths,
                llm_client=llm_client,
                document=document,
                ingest_run_id=ingest_run_id,
                corpus_snapshot=snapshot_id,
            )
        )
    _write_wiki_snapshot_manifest(paths.wiki_dir, snapshot_id, corpus_order=corpus_order_token(batch))
    return summaries


def run_wiki_queries(
    config: AppConfig,
    paths: ProjectPaths,
    query_cases: list[QueryCase],
    corpus_snapshot: str | None = None,
) -> list[GenerationResult]:
    """Run query-time answer generation strictly from wiki pages."""
    pages = load_pages(paths.wiki_dir)
    llm_client = LLMClient(config=config.llm)
    top_k = config.retrieval_top_k()
    snapshot_identity = corpus_snapshot or _resolve_wiki_snapshot_identity(paths)

    results: list[GenerationResult] = []
    for query in query_cases:
        start = perf_counter()
        selected_pages = retrieve_wiki_pages(pages=pages, query=query.question, top_k=top_k)

        prompt = build_wiki_query_prompt(question=query.question, pages=selected_pages)
        llm_response = llm_client.generate_response(prompt, require_token_usage=True)
        answer = llm_response.text
        run_id = _new_run_id(query.query_id)

        run_dir = paths.artifacts_dir / "wiki_runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "selected_pages.json").write_text(
            json.dumps([page.title for page in selected_pages], indent=2),
            encoding="utf-8",
        )
        (run_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
        (run_dir / "answer.txt").write_text(answer, encoding="utf-8")
        (run_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "query_id": query.query_id,
                    "run_id": run_id,
                    "mode": "wiki",
                    "used_context_ids": [page.slug for page in selected_pages],
                    "corpus_snapshot": snapshot_identity,
                    "token_usage": {
                        "prompt_tokens": llm_response.token_usage.prompt_tokens,
                        "completion_tokens": llm_response.token_usage.completion_tokens,
                        "total_tokens": llm_response.token_usage.total_tokens,
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        results.append(
            GenerationResult(
                query_id=query.query_id,
                answer=answer,
                mode="wiki",
                used_context_ids=[page.slug for page in selected_pages],
                run_id=run_id,
                latency_ms=round((perf_counter() - start) * 1000.0, 3),
                artifact_dir=str(run_dir),
                prompt_tokens=llm_response.token_usage.prompt_tokens,
                completion_tokens=llm_response.token_usage.completion_tokens,
                total_tokens=llm_response.token_usage.total_tokens,
            )
        )
    return results
