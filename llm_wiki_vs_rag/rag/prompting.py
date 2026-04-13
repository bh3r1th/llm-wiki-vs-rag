"""Prompt construction for RAG generation."""

from llm_wiki_vs_rag.models import RetrievedChunk


def build_rag_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    """Build synthesis prompt constrained to retrieved context only."""
    context = "\n\n".join(
        (
            f"[chunk_id={chunk.chunk_id} | doc_id={chunk.doc_id} | "
            f"source={chunk.source_path} | pos={chunk.position}]\n{chunk.text}"
        )
        for chunk in chunks
    )

    return (
        "You are answering a question strictly from provided context chunks.\n"
        "Rules:\n"
        "1) Answer only from the context below.\n"
        "2) If context is insufficient, reply exactly: INSUFFICIENT_EVIDENCE.\n"
        "3) Do not invent latest-state, current-status, or time-sensitive claims unless directly supported by context.\n\n"
        f"Question:\n{question}\n\n"
        f"Context:\n{context if context else '[no context provided]'}\n\n"
        "Answer:"
    )
