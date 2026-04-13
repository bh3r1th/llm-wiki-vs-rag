"""Deterministic lexical retrieval from markdown wiki pages."""

from __future__ import annotations

import re

from llm_wiki_vs_rag.wiki.pages import PageRecord


_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text.lower())


def retrieve_wiki_pages(pages: list[PageRecord], query: str, top_k: int = 3) -> list[PageRecord]:
    """Rank pages by transparent keyword overlap over titles and content."""
    query_terms = _tokens(query)
    if not query_terms:
        return pages[:top_k]

    scored = []
    for page in pages:
        title_terms = _tokens(page.title)
        body_terms = _tokens(page.content)
        title_score = sum(title_terms.count(term) for term in query_terms)
        body_score = sum(body_terms.count(term) for term in query_terms)
        total_score = (2 * title_score) + body_score
        scored.append((total_score, page.title.lower(), page))

    scored.sort(key=lambda item: (-item[0], item[1]))
    selected = [page for score, _, page in scored if score > 0]
    if not selected:
        selected = [page for _, _, page in scored]
    return selected[:top_k]
