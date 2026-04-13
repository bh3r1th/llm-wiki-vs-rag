"""Wiki page IO helpers."""

from pathlib import Path

from llm_wiki_vs_rag.models import WikiPage


def persist_pages(pages: list[WikiPage], wiki_dir: Path) -> None:
    """Persist wiki pages into markdown files."""
    wiki_dir.mkdir(parents=True, exist_ok=True)
    for page in pages:
        page_path = wiki_dir / f"{page.page_id}.md"
        page_path.write_text(f"# {page.title}\n\n{page.body}\n", encoding="utf-8")
