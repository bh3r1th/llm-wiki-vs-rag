"""Chunking helpers for RAG indexing."""

from llm_wiki_vs_rag.models import RetrievedChunk, SourceDocument


def chunk_document(
    document: SourceDocument,
    chunk_size_chars: int,
    chunk_overlap_chars: int,
) -> list[RetrievedChunk]:
    """Create deterministic character chunks with overlap."""
    if chunk_size_chars <= 0:
        raise ValueError("chunk_size_chars must be > 0")
    if chunk_overlap_chars < 0:
        raise ValueError("chunk_overlap_chars must be >= 0")
    if chunk_overlap_chars >= chunk_size_chars:
        raise ValueError("chunk_overlap_chars must be < chunk_size_chars")

    text = document.text
    step = chunk_size_chars - chunk_overlap_chars

    chunks: list[RetrievedChunk] = []
    chunk_index = 0
    start = 0
    while start < len(text):
        end = min(start + chunk_size_chars, len(text))
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(
                RetrievedChunk(
                    doc_id=document.doc_id,
                    chunk_id=f"{document.doc_id}:{chunk_index}",
                    text=chunk_text,
                    source_path=document.source_path,
                    position={"chunk_index": chunk_index, "start_char": start, "end_char": end},
                    score=0.0,
                )
            )
            chunk_index += 1
        start += step

    return chunks
