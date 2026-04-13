"""Index construction stubs for RAG."""

from llm_wiki_vs_rag.models import DocumentBatch, RetrievedChunk
from llm_wiki_vs_rag.rag.chunking import chunk_document


def build_in_memory_index(batch: DocumentBatch, chunk_size: int) -> list[RetrievedChunk]:
    """Build a simple list-based index from source documents."""
    indexed_chunks: list[RetrievedChunk] = []
    for document in batch.documents:
        indexed_chunks.extend(chunk_document(document=document, chunk_size=chunk_size))
    return indexed_chunks
