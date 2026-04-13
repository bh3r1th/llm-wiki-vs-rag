"""Pipeline entry points for ingest-time wiki synthesis."""

from llm_wiki_vs_rag.config import AppConfig
from llm_wiki_vs_rag.data.load_docs import load_source_documents
from llm_wiki_vs_rag.llm.client import LLMClient
from llm_wiki_vs_rag.models import GenerationResult, QueryCase, WikiPage
from llm_wiki_vs_rag.paths import ProjectPaths
from llm_wiki_vs_rag.wiki.ingest import synthesize_pages
from llm_wiki_vs_rag.wiki.links import build_page_links
from llm_wiki_vs_rag.wiki.pages import persist_pages
from llm_wiki_vs_rag.wiki.prompting import build_wiki_prompt
from llm_wiki_vs_rag.wiki.retrieve import retrieve_wiki_pages


def ingest_wiki(config: AppConfig, paths: ProjectPaths) -> list[WikiPage]:
    """Create and persist wiki pages from raw documents."""
    batch = load_source_documents(paths.raw_dir)
    pages = synthesize_pages(batch)
    linked_pages = build_page_links(pages, max_links_per_page=config.wiki.max_links_per_page)
    persist_pages(linked_pages, paths.wiki_dir)
    return linked_pages


def run_wiki_queries(
    config: AppConfig,
    paths: ProjectPaths,
    query_cases: list[QueryCase],
) -> list[GenerationResult]:
    """Run queries against ingest-time wiki representation."""
    pages = ingest_wiki(config=config, paths=paths)
    llm_client = LLMClient(config=config.llm)

    results: list[GenerationResult] = []
    for query in query_cases:
        selected_pages = retrieve_wiki_pages(pages=pages, query=query.question, top_k=3)
        prompt = build_wiki_prompt(question=query.question, pages=selected_pages)
        answer = llm_client.generate(prompt)
        results.append(
            GenerationResult(
                query_id=query.query_id,
                answer=answer,
                mode="wiki",
                used_context_ids=[page.page_id for page in selected_pages],
            )
        )
    return results
