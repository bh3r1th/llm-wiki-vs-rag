"""Ingest pipeline primitives for sequential wiki synthesis."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from llm_wiki_vs_rag.llm.client import LLMClient
from llm_wiki_vs_rag.models import SourceDocument
from llm_wiki_vs_rag.paths import ProjectPaths
from llm_wiki_vs_rag.wiki.pages import (
    append_log,
    create_page,
    load_pages,
    rebuild_index,
    update_page_non_destructive,
)
from llm_wiki_vs_rag.wiki.prompting import build_ingest_prompt, coerce_ingest_output
from llm_wiki_vs_rag.wiki.retrieve import retrieve_wiki_pages


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def ingest_one_document(paths: ProjectPaths, llm_client: LLMClient, document: SourceDocument) -> dict:
    """Ingest one raw document against existing wiki state."""
    timestamp = _utc_timestamp()
    current_pages = load_pages(paths.wiki_dir)
    selected_pages = retrieve_wiki_pages(pages=current_pages, query=document.text, top_k=5)

    prompt = build_ingest_prompt(document=document, selected_pages=selected_pages)
    llm_raw = llm_client.generate_json(prompt)
    llm_output = coerce_ingest_output(raw_output=llm_raw, document=document)

    created_titles: list[str] = []
    updated_titles: list[str] = []

    for entry in llm_output["pages_to_create"]:
        title = str(entry.get("title", "")).strip() or document.doc_id
        summary = str(entry.get("summary", "")).strip() or f"Facts from {document.doc_id}."
        content = str(entry.get("content", "")).strip() or document.text
        create_page(
            wiki_dir=paths.wiki_dir,
            title=title,
            summary=summary,
            content=content,
            timestamp=timestamp,
            doc_id=document.doc_id,
        )
        created_titles.append(title)

    for entry in llm_output["pages_to_update"]:
        title = str(entry.get("title", "")).strip()
        if not title:
            continue
        content = str(entry.get("content", "")).strip() or document.text
        change_note = str(entry.get("change_note", "")).strip() or "Updated from new source information."
        update_page_non_destructive(
            wiki_dir=paths.wiki_dir,
            title=title,
            content=content,
            change_note=change_note,
            timestamp=timestamp,
            doc_id=document.doc_id,
        )
        updated_titles.append(title)

    all_pages = load_pages(paths.wiki_dir)
    rebuild_index(index_path=paths.index_md, pages=all_pages, index_note=llm_output["index_note"])
    append_log(
        log_path=paths.log_md,
        timestamp=timestamp,
        doc_id=document.doc_id,
        pages_created=created_titles,
        pages_updated=updated_titles,
        log_note=llm_output["log_note"],
    )

    ingest_artifact_dir = paths.artifacts_dir / "wiki_ingest" / document.doc_id
    ingest_artifact_dir.mkdir(parents=True, exist_ok=True)
    _write_json(ingest_artifact_dir / "selected_pages.json", [page.title for page in selected_pages])
    (ingest_artifact_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    _write_json(ingest_artifact_dir / "llm_output.json", llm_output)
    _write_json(
        ingest_artifact_dir / "applied_changes.json",
        {
            "timestamp": timestamp,
            "doc_id": document.doc_id,
            "pages_created": created_titles,
            "pages_updated": updated_titles,
        },
    )

    return {
        "doc_id": document.doc_id,
        "pages_created": created_titles,
        "pages_updated": updated_titles,
    }
