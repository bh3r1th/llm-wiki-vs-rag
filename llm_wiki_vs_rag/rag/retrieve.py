"""Retrieval utilities for RAG."""

import numpy as np

from llm_wiki_vs_rag.models import RetrievedChunk
from llm_wiki_vs_rag.rag.indexing import RAGIndex, embed_query


def retrieve_top_k(index: RAGIndex, query: str, top_k: int) -> list[RetrievedChunk]:
    """Retrieve top-k chunks via cosine similarity in embedding space."""
    if top_k <= 0 or not index.chunks:
        return []

    query_vector = embed_query(query)
    query_norm = float(np.linalg.norm(query_vector))
    if query_norm > 0:
        query_vector = query_vector / query_norm

    chunk_vectors = index.embeddings
    row_norms = np.linalg.norm(chunk_vectors, axis=1, keepdims=True)
    row_norms[row_norms == 0.0] = 1.0
    chunk_vectors = chunk_vectors / row_norms

    scores = np.dot(chunk_vectors, query_vector)
    ordered = np.argsort(-scores)

    results: list[RetrievedChunk] = []
    seen_chunk_ids: set[str] = set()
    for idx in ordered:
        chunk = index.chunks[int(idx)]
        if chunk.chunk_id in seen_chunk_ids:
            continue
        seen_chunk_ids.add(chunk.chunk_id)
        results.append(chunk.model_copy(update={"score": float(scores[int(idx)])}))
        if len(results) >= top_k:
            break

    return results
