"""Focused tests for markdown wiki page handling."""

from llm_wiki_vs_rag.wiki.pages import (
    append_log,
    create_page,
    load_pages,
    rebuild_index,
    slugify,
    update_page_non_destructive,
)


def test_slug_and_page_creation(tmp_path):
    wiki_dir = tmp_path / "wiki"
    record = create_page(
        wiki_dir=wiki_dir,
        title="Alpha Topic!",
        summary="One-line summary",
        content="Initial state",
        timestamp="2026-04-13T00:00:00Z",
        doc_id="001",
    )

    assert slugify("Alpha Topic!") == "alpha-topic"
    assert (wiki_dir / "alpha-topic.md").exists()
    assert record.title == "Alpha Topic!"


def test_index_rebuild_lists_page_summaries(tmp_path):
    wiki_dir = tmp_path / "wiki"
    create_page(
        wiki_dir=wiki_dir,
        title="First",
        summary="First summary",
        content="A",
        timestamp="2026-04-13T00:00:00Z",
        doc_id="001",
    )
    create_page(
        wiki_dir=wiki_dir,
        title="Second",
        summary="Second summary",
        content="B",
        timestamp="2026-04-13T00:00:00Z",
        doc_id="002",
    )

    index_path = tmp_path / "index.md"
    rebuild_index(index_path=index_path, pages=load_pages(wiki_dir), index_note="refresh")

    index_text = index_path.read_text(encoding="utf-8")
    assert "[[First]] — First summary" in index_text
    assert "[[Second]] — Second summary" in index_text


def test_log_append_creates_chronological_entries(tmp_path):
    log_path = tmp_path / "log.md"
    append_log(
        log_path=log_path,
        timestamp="2026-04-13T00:00:00Z",
        doc_id="001",
        pages_created=["First"],
        pages_updated=[],
        log_note="created first page",
    )
    append_log(
        log_path=log_path,
        timestamp="2026-04-13T01:00:00Z",
        doc_id="002",
        pages_created=[],
        pages_updated=["First"],
        log_note="updated first page",
    )

    log_text = log_path.read_text(encoding="utf-8")
    assert "## 2026-04-13T00:00:00Z — 001" in log_text
    assert "## 2026-04-13T01:00:00Z — 002" in log_text


def test_non_destructive_update_keeps_history(tmp_path):
    wiki_dir = tmp_path / "wiki"
    create_page(
        wiki_dir=wiki_dir,
        title="Timeline",
        summary="Summary",
        content="Old facts.",
        timestamp="2026-04-13T00:00:00Z",
        doc_id="001",
    )
    update_page_non_destructive(
        wiki_dir=wiki_dir,
        title="Timeline",
        content="New facts.",
        change_note="Policy changed",
        timestamp="2026-04-13T02:00:00Z",
        doc_id="002",
    )

    page_text = (wiki_dir / "timeline.md").read_text(encoding="utf-8")
    assert "## Current State\nNew facts." in page_text
    assert "Created from raw doc `001`" in page_text
    assert "Source `002`: Policy changed" in page_text
