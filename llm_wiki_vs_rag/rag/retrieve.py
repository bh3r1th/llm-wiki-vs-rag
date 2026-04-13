"""Retrieval utilities for RAG."""

import numpy as np

from llm_wiki_vs_rag.models import RetrievedChunk
from llm_wiki_vs_rag.rag.indexing import RAGIndex, embed_query


def retrieve_top_k(index: RAGIndex, query: str, top_k: int) -> list[RetrievedChunk]:
    """Retrieve top-k chunks via cosine similarity in embedding space."""
    if top_k <= 0 or not index.chunks:
        return []

    query_vector = embed_query(query)
    scores = np.dot(index.embeddings, query_vector)
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
