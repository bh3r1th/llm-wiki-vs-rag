"""Prompt builders for wiki-based generation."""

from llm_wiki_vs_rag.models import WikiPage


def build_wiki_prompt(question: str, pages: list[WikiPage]) -> str:
    """Build generation prompt using retrieved wiki pages."""
    context = "\n\n".join(f"[{page.page_id}] {page.body}" for page in pages)
    return f"Use the wiki pages to answer the question.\n\nQuestion: {question}\n\nPages:\n{context}"
