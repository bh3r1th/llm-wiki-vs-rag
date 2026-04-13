"""Prompt builders for ingest-time wiki synthesis and wiki query answering."""

from __future__ import annotations

import json

from llm_wiki_vs_rag.models import SourceDocument
from llm_wiki_vs_rag.wiki.pages import PageRecord


def build_ingest_prompt(document: SourceDocument, selected_pages: list[PageRecord]) -> str:
    """Build strict JSON-only prompt for conservative wiki updates."""
    context_pages = []
    for page in selected_pages:
        context_pages.append(f"### {page.title}\n{page.content}")

    page_context = "\n\n".join(context_pages) if context_pages else "(No existing relevant wiki pages.)"

    schema = {
        "pages_to_create": [{"title": "", "summary": "", "content": ""}],
        "pages_to_update": [{"title": "", "content": "", "change_note": ""}],
        "index_note": "",
        "log_note": "",
    }

    return (
        "You are updating a filesystem markdown wiki one document at a time.\n"
        "Follow these rules strictly:\n"
        "1) Be conservative: preserve prior valid information unless this new document changes it.\n"
        "2) If information changes, record deprecations/changes in History / Changes, do not erase history.\n"
        "3) Keep content human-readable markdown and add wikilinks like [[Page Name]] for related concepts.\n"
        "4) Return STRICT JSON only (no markdown fences, no prose).\n"
        f"JSON schema example: {json.dumps(schema)}\n\n"
        f"Incoming raw document (doc_id={document.doc_id}):\n{document.text}\n\n"
        f"Relevant existing wiki pages:\n{page_context}\n"
    )


def coerce_ingest_output(raw_output: dict, document: SourceDocument) -> dict:
    """Coerce LLM JSON output into expected strict structure with safe fallback."""
    pages_to_create = raw_output.get("pages_to_create") if isinstance(raw_output, dict) else None
    pages_to_update = raw_output.get("pages_to_update") if isinstance(raw_output, dict) else None

    if not isinstance(pages_to_create, list):
        pages_to_create = []
    if not isinstance(pages_to_update, list):
        pages_to_update = []

    if not pages_to_create and not pages_to_update:
        pages_to_create = [
            {
                "title": document.doc_id.replace("_", " ").title(),
                "summary": f"Facts synthesized from raw source {document.doc_id}.",
                "content": document.text,
            }
        ]

    return {
        "pages_to_create": pages_to_create,
        "pages_to_update": pages_to_update,
        "index_note": (raw_output.get("index_note") if isinstance(raw_output, dict) else "") or "",
        "log_note": (raw_output.get("log_note") if isinstance(raw_output, dict) else "") or "",
    }


def build_wiki_query_prompt(question: str, pages: list[PageRecord]) -> str:
    """Build generation prompt using retrieved wiki pages."""
    context = "\n\n".join(f"[{page.title}]\n{page.content}" for page in pages)
    return (
        "Use the wiki pages below to answer the question. Cite concrete facts from these pages only.\n\n"
        f"Question: {question}\n\n"
        f"Wiki context:\n{context if context else '(No wiki pages retrieved.)'}"
    )
