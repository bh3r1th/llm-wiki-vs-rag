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
    """Coerce LLM JSON output into expected strict structure with strict validation."""
    if not isinstance(raw_output, dict):
        raise ValueError(f"Ingest model output must be a JSON object for doc_id={document.doc_id}.")
    pages_to_create = raw_output.get("pages_to_create")
    pages_to_update = raw_output.get("pages_to_update")
    if not isinstance(pages_to_create, list) or not isinstance(pages_to_update, list):
        raise ValueError(
            f"Ingest model output must include list fields pages_to_create/pages_to_update for doc_id={document.doc_id}."
        )

    normalized_create: list[dict[str, str]] = []
    for entry in pages_to_create:
        if not isinstance(entry, dict):
            raise ValueError(f"Invalid pages_to_create entry type for doc_id={document.doc_id}.")
        title = str(entry.get("title", "")).strip()
        summary = str(entry.get("summary", "")).strip()
        content = str(entry.get("content", "")).strip()
        if not title or not summary or not content:
            raise ValueError(
                f"Invalid pages_to_create entry: title, summary, and content are required for doc_id={document.doc_id}."
            )
        normalized_create.append({"title": title, "summary": summary, "content": content})

    normalized_update: list[dict[str, str]] = []
    for entry in pages_to_update:
        if not isinstance(entry, dict):
            raise ValueError(f"Invalid pages_to_update entry type for doc_id={document.doc_id}.")
        title = str(entry.get("title", "")).strip()
        content = str(entry.get("content", "")).strip()
        change_note = str(entry.get("change_note", "")).strip()
        if not title or not content:
            raise ValueError(
                f"Invalid pages_to_update entry: title and content are required for doc_id={document.doc_id}."
            )
        normalized_update.append(
            {
                "title": title,
                "content": content,
                "change_note": change_note or "Updated from new source information.",
            }
        )

    if not normalized_create and not normalized_update:
        raise ValueError(f"Ingest model output is empty for doc_id={document.doc_id}; no wiki changes proposed.")

    return {
        "pages_to_create": normalized_create,
        "pages_to_update": normalized_update,
        "index_note": str(raw_output.get("index_note", "")).strip(),
        "log_note": str(raw_output.get("log_note", "")).strip(),
    }


def build_wiki_query_prompt(question: str, pages: list[PageRecord]) -> str:
    """Build generation prompt using retrieved wiki pages."""
    context = "\n\n".join(f"[{page.title}]\n{page.content}" for page in pages)
    return (
        "You are answering a question strictly from provided wiki context.\n"
        "Rules:\n"
        "1) Answer only from the wiki context below.\n"
        "2) If context is insufficient, reply exactly: INSUFFICIENT_EVIDENCE.\n"
        "3) Do not invent latest-state, current-status, or time-sensitive claims unless directly supported by context.\n\n"
        f"Question:\n{question}\n\n"
        f"Wiki context:\n{context if context else '[no context provided]'}\n\n"
        "Answer:"
    )
