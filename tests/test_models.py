"""Tests for core model validation basics."""

import pytest
from pydantic import ValidationError

from llm_wiki_vs_rag.models import QueryCase, SourceDocument


def test_source_document_validates_required_fields(tmp_path):
    doc = SourceDocument(doc_id="doc-1", source_path=tmp_path / "a.txt", text="content")
    assert doc.doc_id == "doc-1"


def test_query_case_rejects_empty_question():
    with pytest.raises(ValidationError):
        QueryCase(query_id="q1", question="")
