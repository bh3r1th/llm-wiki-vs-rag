"""Pipeline entry points for RAG operations."""

from llm_wiki_vs_rag.config import AppConfig
from llm_wiki_vs_rag.data.load_docs import load_source_documents
from llm_wiki_vs_rag.llm.client import LLMClient
from llm_wiki_vs_rag.models import GenerationResult, QueryCase
from llm_wiki_vs_rag.paths import ProjectPaths
from llm_wiki_vs_rag.rag.indexing import build_in_memory_index
from llm_wiki_vs_rag.rag.prompting import build_rag_prompt
from llm_wiki_vs_rag.rag.retrieve import retrieve_top_k


def build_rag_index(config: AppConfig, paths: ProjectPaths) -> list:
    """Build a minimal in-memory RAG index from raw documents."""
    batch = load_source_documents(paths.raw_dir)
    return build_in_memory_index(batch=batch, chunk_size=config.rag.chunk_size)


def run_rag_queries(
    config: AppConfig,
    paths: ProjectPaths,
    query_cases: list[QueryCase],
) -> list[GenerationResult]:
    """Run query-time synthesis for provided query cases."""
    index = build_rag_index(config=config, paths=paths)
    llm_client = LLMClient(config=config.llm)

    results: list[GenerationResult] = []
    for query in query_cases:
        chunks = retrieve_top_k(index=index, query=query.question, top_k=config.rag.top_k)
        prompt = build_rag_prompt(question=query.question, chunks=chunks)
        answer = llm_client.generate(prompt)
        results.append(
            GenerationResult(
                query_id=query.query_id,
                answer=answer,
                mode="rag",
                used_context_ids=[chunk.chunk_id for chunk in chunks],
            )
        )
    return results
