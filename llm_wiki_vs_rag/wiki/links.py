"""Wiki link helpers for markdown wikilinks."""

from __future__ import annotations

import re

WIKILINK_PATTERN = re.compile(r"\[\[([^\[\]|]+)(?:\|[^\]]+)?\]\]")


def extract_wikilinks(markdown_text: str) -> list[str]:
    """Extract unique wikilink targets from markdown in appearance order."""
    seen: set[str] = set()
    links: list[str] = []
    for match in WIKILINK_PATTERN.finditer(markdown_text):
        target = match.group(1).strip()
        if not target or target in seen:
            continue
        seen.add(target)
        links.append(target)
    return links


def ensure_related_links_section(markdown_text: str, links: list[str]) -> str:
    """Ensure markdown has a `Related Pages` section reflecting wikilinks."""
    if not links:
        return markdown_text

    section_header = "## Related Pages"
    link_lines = "\n".join(f"- [[{link}]]" for link in links)

    if section_header not in markdown_text:
        suffix = "\n\n" if markdown_text.strip() else ""
        return f"{markdown_text.rstrip()}{suffix}{section_header}\n{link_lines}\n"

    pattern = re.compile(r"(?ms)^## Related Pages\n.*?(?=\n## |\Z)")
    replacement = f"## Related Pages\n{link_lines}\n"
    return pattern.sub(replacement, markdown_text, count=1)
