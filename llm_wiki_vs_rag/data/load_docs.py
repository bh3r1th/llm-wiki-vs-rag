"""Document loading from filesystem."""

from pathlib import Path

from llm_wiki_vs_rag.models import DocumentBatch, SourceDocument


def load_source_documents(raw_dir: Path) -> DocumentBatch:
    """Load .txt files from raw directory into typed source documents."""
    documents: list[SourceDocument] = []
    for file_path in sorted(raw_dir.glob("*.txt")):
        text = file_path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        documents.append(
            SourceDocument(
                doc_id=file_path.stem,
                source_path=file_path,
                text=text,
                metadata={"filename": file_path.name},
            )
        )
    return DocumentBatch(documents=documents)
