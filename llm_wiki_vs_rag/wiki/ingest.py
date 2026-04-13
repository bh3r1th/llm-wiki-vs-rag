"""Ingest-time synthesis primitives for wiki pages."""

from llm_wiki_vs_rag.models import DocumentBatch, WikiPage


def synthesize_pages(batch: DocumentBatch) -> list[WikiPage]:
    """Create one wiki page per source document as a minimal baseline."""
    pages: list[WikiPage] = []
    for doc in batch.documents:
        pages.append(WikiPage(page_id=doc.doc_id, title=doc.doc_id, body=doc.text, links=[]))
    return pages
