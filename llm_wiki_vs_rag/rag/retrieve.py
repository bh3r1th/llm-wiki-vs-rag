"""Retrieval stubs for RAG."""

from llm_wiki_vs_rag.models import RetrievedChunk


def retrieve_top_k(index: list[RetrievedChunk], query: str, top_k: int) -> list[RetrievedChunk]:
    """Return top-k chunks using simple lexical overlap placeholder logic."""
    query_terms = {term.lower() for term in query.split() if term}
    rescored: list[RetrievedChunk] = []
    for chunk in index:
        overlap = sum(1 for term in query_terms if term in chunk.text.lower())
        rescored.append(chunk.model_copy(update={"score": float(overlap)}))
    return sorted(rescored, key=lambda item: item.score, reverse=True)[:top_k]
