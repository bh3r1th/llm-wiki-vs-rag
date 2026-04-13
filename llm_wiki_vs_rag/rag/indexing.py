"""RAG vector index construction and persistence."""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from llm_wiki_vs_rag.models import DocumentBatch, RetrievedChunk
from llm_wiki_vs_rag.rag.chunking import chunk_document


INDEX_DIRNAME = "rag_index"
EMBED_DIM = 256


@dataclass
class RAGIndex:
    """In-memory index bundle used for retrieval."""

    chunks: list[RetrievedChunk]
    embeddings: np.ndarray
    backend: str


def _embed_text(text: str, dim: int = EMBED_DIM) -> np.ndarray:
    """Create a deterministic local embedding using hashed token counts."""
    vector = np.zeros(dim, dtype=np.float32)
    for token in text.lower().split():
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        idx = int.from_bytes(digest, byteorder="big") % dim
        vector[idx] += 1.0
    return _normalize_vector(vector)


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    """Normalize one embedding vector for exact cosine scoring."""
    norm = float(np.linalg.norm(vector))
    if norm > 0:
        vector /= norm
    return vector


def _normalize_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Normalize each embedding row for exact cosine scoring."""
    if embeddings.size == 0:
        return embeddings.astype(np.float32)
    row_norms = np.linalg.norm(embeddings, axis=1, keepdims=True).astype(np.float32)
    row_norms[row_norms == 0.0] = 1.0
    return (embeddings / row_norms).astype(np.float32)


def _embed_texts(texts: list[str], dim: int = EMBED_DIM) -> np.ndarray:
    return _normalize_matrix(np.vstack([_embed_text(text, dim=dim) for text in texts]).astype(np.float32))


def build_in_memory_index(
    batch: DocumentBatch,
    chunk_size_chars: int,
    chunk_overlap_chars: int,
) -> RAGIndex:
    """Build a simple local vector index over deterministic document chunks."""
    chunks: list[RetrievedChunk] = []
    for document in batch.documents:
        chunks.extend(
            chunk_document(
                document=document,
                chunk_size_chars=chunk_size_chars,
                chunk_overlap_chars=chunk_overlap_chars,
            )
        )

    embeddings = _embed_texts([chunk.text for chunk in chunks]) if chunks else np.zeros((0, EMBED_DIM), dtype=np.float32)

    return RAGIndex(chunks=chunks, embeddings=embeddings, backend="numpy")


def persist_index(
    index: RAGIndex,
    artifacts_dir: Path,
    snapshot_id: str,
    execution_fingerprint: str,
    corpus_order: str | None = None,
) -> Path:
    """Persist vector matrix and chunk metadata under artifacts/rag_index."""
    index_dir = artifacts_dir / INDEX_DIRNAME
    index_dir.mkdir(parents=True, exist_ok=True)

    normalized_embeddings = _normalize_matrix(index.embeddings.astype(np.float32))
    np.save(index_dir / "embeddings.npy", normalized_embeddings)
    metadata_path = index_dir / "chunk_metadata.jsonl"
    with metadata_path.open("w", encoding="utf-8") as handle:
        for chunk in index.chunks:
            handle.write(chunk.model_dump_json())
            handle.write("\n")

    manifest = {
        "backend": "numpy",
        "embedding_dim": int(normalized_embeddings.shape[1]) if normalized_embeddings.ndim == 2 else EMBED_DIM,
        "num_chunks": len(index.chunks),
        "snapshot_id": snapshot_id,
        "execution_fingerprint": execution_fingerprint,
        "corpus_order": corpus_order,
    }
    (index_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return index_dir


def load_index(artifacts_dir: Path) -> RAGIndex:
    """Load previously persisted RAG index artifacts."""
    index_dir = artifacts_dir / INDEX_DIRNAME
    embeddings = _normalize_matrix(np.load(index_dir / "embeddings.npy").astype(np.float32))

    chunks: list[RetrievedChunk] = []
    with (index_dir / "chunk_metadata.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                chunks.append(RetrievedChunk.model_validate_json(line))

    manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
    return RAGIndex(chunks=chunks, embeddings=embeddings, backend=manifest.get("backend", "numpy"))


def embed_query(query: str) -> np.ndarray:
    """Embed a retrieval query into the same vector space as chunks."""
    return _embed_text(query)
