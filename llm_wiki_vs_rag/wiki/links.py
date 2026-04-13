"""Wiki link helpers."""

from llm_wiki_vs_rag.models import WikiPage


def build_page_links(pages: list[WikiPage], max_links_per_page: int) -> list[WikiPage]:
    """Assign deterministic links between pages."""
    page_ids = [page.page_id for page in pages]
    updated: list[WikiPage] = []
    for page in pages:
        links = [pid for pid in page_ids if pid != page.page_id][:max_links_per_page]
        updated.append(page.model_copy(update={"links": links}))
    return updated
