"""Document loading from filesystem."""

import hashlib
import json
import re
from pathlib import Path

from llm_wiki_vs_rag.models import DocumentBatch, SourceDocument


def load_source_documents(raw_dir: Path) -> DocumentBatch:
    """Load .txt/.md files from raw directory with explicit validated chronology."""
    documents: list[SourceDocument] = []
    candidate_paths = sorted(
        [*raw_dir.glob("*.txt"), *raw_dir.glob("*.md")],
        key=lambda path: path.name,
    )
    chronology: list[tuple[int, int, str]] = []
    for file_path in candidate_paths:
        match = re.match(r"^(\d+)(?:[_-].*)?$", file_path.stem)
        if match is None:
            raise ValueError(
                "Raw document chronology requires filename stems to start with a zero-padded numeric prefix "
                f"(example: 001_event.md). invalid={file_path.name}"
            )
        prefix = match.group(1)
        chronology.append((int(prefix), len(prefix), file_path.name))
    widths = {width for _, width, _ in chronology}
    if len(widths) > 1:
        raise ValueError(
            "Raw document chronology requires a consistent numeric prefix width across all files. "
            f"widths={sorted(widths)}"
        )
    prefix_counts: dict[int, int] = {}
    for value, _, _ in chronology:
        prefix_counts[value] = prefix_counts.get(value, 0) + 1
    duplicate_prefixes = sorted(value for value, count in prefix_counts.items() if count > 1)
    if duplicate_prefixes:
        raise ValueError(
            "Raw document chronology requires unique numeric prefixes. "
            f"duplicate_prefixes={duplicate_prefixes[:5]}"
        )
    if chronology != sorted(chronology, key=lambda item: item[0]):
        raise ValueError(
            "Raw document chronology validation failed: filename lexicographic order does not match numeric prefix order."
        )

    for file_path in candidate_paths:
        text = file_path.read_text(encoding="utf-8")
        documents.append(
            SourceDocument(
                doc_id=file_path.stem,
                source_path=file_path,
                text=text,
                metadata={"filename": file_path.name, "suffix": file_path.suffix.lower()},
            )
        )

    return DocumentBatch(
        documents=documents,
        chronology=[
            {"position": value, "prefix_width": width, "filename": filename}
            for value, width, filename in chronology
        ],
    )


def fingerprint_document_batch(batch: DocumentBatch) -> str:
    """Compute deterministic corpus snapshot identity from loaded document contents."""
    canonical_documents = sorted(
        [{"doc_id": document.doc_id, "text": document.text} for document in batch.documents],
        key=lambda item: item["doc_id"],
    )
    payload = json.dumps(canonical_documents, ensure_ascii=False, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def corpus_order_token(batch: DocumentBatch) -> str:
    """Return explicit corpus chronology token from validated batch chronology."""
    if not batch.chronology:
        raise ValueError("Cannot derive corpus chronology token from an empty raw corpus.")
    last_position = int(batch.chronology[-1]["position"])
    width = int(batch.chronology[-1]["prefix_width"])
    return f"{last_position:0{width}d}"
