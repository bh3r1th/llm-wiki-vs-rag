"""Pipeline entry points for ingest-time wiki synthesis and querying."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from llm_wiki_vs_rag.config import AppConfig
from llm_wiki_vs_rag.data.load_docs import load_source_documents
from llm_wiki_vs_rag.llm.client import LLMClient
from llm_wiki_vs_rag.models import GenerationResult, QueryCase
from llm_wiki_vs_rag.paths import ProjectPaths
from llm_wiki_vs_rag.rag.pipeline import answer_rag_query
from llm_wiki_vs_rag.wiki.ingest import ingest_one_document
from llm_wiki_vs_rag.wiki.pages import load_pages
from llm_wiki_vs_rag.wiki.prompting import build_wiki_query_prompt
from llm_wiki_vs_rag.wiki.retrieve import retrieve_wiki_pages


def _new_run_id(query_id: str) -> str:
    return f"{query_id}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"


def ingest_wiki(config: AppConfig, paths: ProjectPaths):
    """Process raw docs sequentially and update markdown wiki incrementally."""
    batch = load_source_documents(paths.raw_dir)
    llm_client = LLMClient(config=config.llm)
    summaries = []
    for document in batch.documents:
        summaries.append(ingest_one_document(paths=paths, llm_client=llm_client, document=document))
    return summaries


def run_wiki_queries(
    config: AppConfig,
    paths: ProjectPaths,
    query_cases: list[QueryCase],
    use_rag_fallback: bool = False,
) -> list[GenerationResult]:
    """Run query-time answer generation from wiki pages with optional RAG fallback."""
    pages = load_pages(paths.wiki_dir)
    llm_client = LLMClient(config=config.llm)

    results: list[GenerationResult] = []
    for query in query_cases:
        selected_pages = retrieve_wiki_pages(pages=pages, query=query.question, top_k=3)

        if use_rag_fallback and not selected_pages:
            results.append(answer_rag_query(config=config, paths=paths, query=query))
            continue

        prompt = build_wiki_query_prompt(question=query.question, pages=selected_pages)
        answer = llm_client.generate(prompt)
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
                    "fallback_to_rag": use_rag_fallback,
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
            )
        )
    return results
