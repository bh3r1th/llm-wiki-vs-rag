"""Retrieval from wiki pages."""

from llm_wiki_vs_rag.models import WikiPage


def retrieve_wiki_pages(pages: list[WikiPage], query: str, top_k: int = 3) -> list[WikiPage]:
    """Return top-k pages using naive lexical overlap scoring."""
    query_terms = {term.lower() for term in query.split() if term}
    scored = sorted(
        pages,
        key=lambda page: sum(1 for term in query_terms if term in page.body.lower()),
        reverse=True,
    )
    return scored[:top_k]
