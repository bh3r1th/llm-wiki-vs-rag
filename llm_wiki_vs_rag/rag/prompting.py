"""Prompt construction for RAG generation."""

from llm_wiki_vs_rag.models import RetrievedChunk


def build_rag_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    """Build a compact prompt from query and retrieved chunks."""
    context = "\n\n".join(f"[{chunk.chunk_id}] {chunk.text}" for chunk in chunks)
    return f"Answer the question using the context.\n\nQuestion: {question}\n\nContext:\n{context}"
