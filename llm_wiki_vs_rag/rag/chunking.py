"""Chunking helpers for RAG indexing."""

from llm_wiki_vs_rag.models import RetrievedChunk, SourceDocument


def chunk_document(document: SourceDocument, chunk_size: int) -> list[RetrievedChunk]:
    """Create fixed-size character chunks from a source document."""
    text = document.text
    chunks: list[RetrievedChunk] = []
    for idx in range(0, len(text), chunk_size):
        chunk_text = text[idx : idx + chunk_size].strip()
        if not chunk_text:
            continue
        chunks.append(
            RetrievedChunk(
                doc_id=document.doc_id,
                chunk_id=f"{document.doc_id}:{idx // chunk_size}",
                text=chunk_text,
                score=0.0,
            )
        )
    return chunks
