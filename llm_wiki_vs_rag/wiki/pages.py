"""Wiki page models and markdown file handling."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

from llm_wiki_vs_rag.wiki.links import ensure_related_links_section, extract_wikilinks


@dataclass(slots=True)
class PageRecord:
    """In-memory representation of one markdown wiki page."""

    title: str
    slug: str
    path: Path
    summary: str
    content: str


def slugify(value: str) -> str:
    """Create deterministic slug suitable for markdown filenames."""
    normalized = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return normalized or "untitled"


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", delete=False, dir=str(path.parent), encoding="utf-8") as handle:
        handle.write(text)
        tmp_path = Path(handle.name)
    tmp_path.replace(path)


def page_path_for_title(wiki_dir: Path, title: str) -> Path:
    return wiki_dir / f"{slugify(title)}.md"


def list_page_files(wiki_dir: Path) -> list[Path]:
    if not wiki_dir.exists():
        return []
    return sorted(wiki_dir.glob("*.md"), key=lambda path: path.name)


def read_page(page_path: Path) -> PageRecord:
    raw = page_path.read_text(encoding="utf-8").strip()
    lines = raw.splitlines()
    title = page_path.stem.replace("-", " ").title()
    if lines and lines[0].startswith("# "):
        title = lines[0][2:].strip()

    summary = ""
    for line in lines[1:]:
        if line.startswith("Summary:"):
            summary = line.split(":", 1)[1].strip()
            break

    return PageRecord(
        title=title,
        slug=page_path.stem,
        path=page_path,
        summary=summary,
        content=raw,
    )


def load_pages(wiki_dir: Path) -> list[PageRecord]:
    return [read_page(path) for path in list_page_files(wiki_dir)]


def _render_page(title: str, summary: str, current_state: str, history_line: str) -> str:
    return (
        f"# {title}\n\n"
        f"Summary: {summary.strip()}\n\n"
        "## Current State\n"
        f"{current_state.strip()}\n\n"
        "## History / Changes\n"
        f"- {history_line.strip()}\n\n"
        "## Related Pages\n"
    ).rstrip() + "\n"


def create_page(wiki_dir: Path, title: str, summary: str, content: str, timestamp: str, doc_id: str) -> PageRecord:
    """Create a markdown page using a human-readable template."""
    page_path = page_path_for_title(wiki_dir=wiki_dir, title=title)
    history_line = f"{timestamp} — Created from raw doc `{doc_id}`."
    rendered = _render_page(title=title, summary=summary, current_state=content, history_line=history_line)
    rendered = ensure_related_links_section(rendered, extract_wikilinks(rendered))
    _atomic_write(page_path, rendered)
    return read_page(page_path)


def _replace_current_state(markdown_text: str, new_content: str) -> str:
    pattern = re.compile(r"(?ms)^## Current State\n.*?(?=\n## |\Z)")
    replacement = f"## Current State\n{new_content.strip()}\n"
    if pattern.search(markdown_text):
        return pattern.sub(replacement, markdown_text, count=1)
    return f"{markdown_text.rstrip()}\n\n{replacement}"


def _append_history(markdown_text: str, history_line: str) -> str:
    pattern = re.compile(r"(?ms)^## History / Changes\n.*?(?=\n## |\Z)")
    match = pattern.search(markdown_text)
    if not match:
        return f"{markdown_text.rstrip()}\n\n## History / Changes\n- {history_line}\n"

    section = match.group(0).rstrip()
    if not section.endswith("\n"):
        section += "\n"
    section += f"- {history_line}\n"
    return f"{markdown_text[:match.start()]}{section}{markdown_text[match.end():]}"


def update_page_non_destructive(
    wiki_dir: Path,
    title: str,
    content: str,
    change_note: str,
    timestamp: str,
    doc_id: str,
) -> PageRecord:
    """Update page current state while retaining prior history and sections."""
    page_path = page_path_for_title(wiki_dir=wiki_dir, title=title)
    if not page_path.exists():
        return create_page(
            wiki_dir=wiki_dir,
            title=title,
            summary=f"Created during update from {doc_id}",
            content=content,
            timestamp=timestamp,
            doc_id=doc_id,
        )

    markdown = page_path.read_text(encoding="utf-8")
    markdown = _replace_current_state(markdown, content)
    history_line = f"{timestamp} — Source `{doc_id}`: {change_note.strip()}"
    markdown = _append_history(markdown, history_line=history_line)
    markdown = ensure_related_links_section(markdown, extract_wikilinks(markdown))
    _atomic_write(page_path, markdown)
    return read_page(page_path)


def rebuild_index(index_path: Path, pages: list[PageRecord], index_note: str = "") -> None:
    """Rebuild `index.md` listing all pages with one-line descriptions."""
    lines = ["# Wiki Index", ""]
    if index_note.strip():
        lines.append(f"Note: {index_note.strip()}")
        lines.append("")

    for page in sorted(pages, key=lambda record: record.title.lower()):
        summary = page.summary or "No summary provided."
        lines.append(f"- [[{page.title}]] — {summary}")

    _atomic_write(index_path, "\n".join(lines).rstrip() + "\n")


def append_log(log_path: Path, timestamp: str, doc_id: str, pages_created: list[str], pages_updated: list[str], log_note: str) -> None:
    """Append one chronological markdown log entry for each ingested doc."""
    if not log_path.exists() or not log_path.read_text(encoding="utf-8").strip():
        base = "# Wiki Ingest Log\n\n"
    else:
        base = log_path.read_text(encoding="utf-8").rstrip() + "\n\n"

    base += f"## {timestamp} — {doc_id}\n"
    base += f"- pages_created: {', '.join(pages_created) if pages_created else '(none)'}\n"
    base += f"- pages_updated: {', '.join(pages_updated) if pages_updated else '(none)'}\n"
    base += f"- note: {log_note.strip() or '(none)'}\n"
    _atomic_write(log_path, base)
