"""Helpers for strict benchmark reproducibility identities and validation."""

from __future__ import annotations

import hashlib
import json

from llm_wiki_vs_rag.config import AppConfig
from llm_wiki_vs_rag.data.load_docs import fingerprint_document_batch, load_source_documents
from llm_wiki_vs_rag.llm.client import resolve_runtime_llm_settings
from llm_wiki_vs_rag.paths import ProjectPaths


RAG_EMBEDDING_METHOD_ID = "local-hash-blake2b-embeddings:v1"
RAG_RETRIEVAL_IMPL_ID = "rag.numpy_cosine:v1"
WIKI_RETRIEVAL_IMPL_ID = "wiki.term_frequency_overlap:v1"
RAG_PROMPT_TEMPLATE_ID = "rag_query_prompt:v1"
WIKI_PROMPT_TEMPLATE_ID = "wiki_query_prompt:v1"


def _hash_payload(payload: dict) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"


def compute_execution_fingerprint(config: AppConfig, system: str) -> str:
    """Compute benchmark-critical execution/config identity."""
    if system not in {"rag", "wiki"}:
        raise ValueError(f"Unsupported system for execution fingerprint: {system}")
    provider, _, _, model_name = resolve_runtime_llm_settings(config.llm)
    payload = {
        "system": system,
        "chunking": {
            "chunk_size": config.rag.chunk_size,
            "chunk_overlap": config.rag.chunk_overlap,
            "top_k": config.retrieval_top_k(),
        },
        "embedding_method_id": RAG_EMBEDDING_METHOD_ID if system == "rag" else "none",
        "retrieval_impl_id": RAG_RETRIEVAL_IMPL_ID if system == "rag" else WIKI_RETRIEVAL_IMPL_ID,
        "model_id": f"{provider}:{model_name}",
        "prompt_template_id": RAG_PROMPT_TEMPLATE_ID if system == "rag" else WIKI_PROMPT_TEMPLATE_ID,
    }
    return _hash_payload(payload)


def validate_current_raw_corpus_snapshot(paths: ProjectPaths, expected_snapshot: str, system: str) -> str:
    """Recompute raw corpus fingerprint at query time and compare to expected manifest snapshot."""
    current_snapshot = fingerprint_document_batch(load_source_documents(paths.raw_dir))
    if current_snapshot != expected_snapshot:
        raise ValueError(
            "Raw corpus snapshot drift detected; query execution refused. "
            f"system={system}, expected={expected_snapshot}, current={current_snapshot}, raw_dir={paths.raw_dir}."
        )
    return current_snapshot
