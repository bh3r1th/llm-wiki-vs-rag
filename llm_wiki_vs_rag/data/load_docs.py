"""Document loading from filesystem."""

import hashlib
import json
from pathlib import Path

from llm_wiki_vs_rag.models import DocumentBatch, SourceDocument


def load_source_documents(raw_dir: Path) -> DocumentBatch:
    """Load .txt/.md files from raw directory in deterministic filename order."""
    documents: list[SourceDocument] = []
    candidate_paths = sorted(
        [*raw_dir.glob("*.txt"), *raw_dir.glob("*.md")],
        key=lambda path: path.name,
    )

    for file_path in candidate_paths:
        text = file_path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        documents.append(
            SourceDocument(
                doc_id=file_path.stem,
                source_path=file_path,
                text=text,
                metadata={"filename": file_path.name, "suffix": file_path.suffix.lower()},
            )
        )

    return DocumentBatch(documents=documents)


def fingerprint_document_batch(batch: DocumentBatch) -> str:
    """Compute deterministic corpus snapshot identity from loaded document contents."""
    canonical_documents = sorted(
        [{"doc_id": document.doc_id, "text": document.text} for document in batch.documents],
        key=lambda item: item["doc_id"],
    )
    payload = json.dumps(canonical_documents, ensure_ascii=False, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"
