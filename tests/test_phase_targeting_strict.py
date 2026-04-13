"""Focused tests for strict phase-targeted benchmark query execution."""

from __future__ import annotations

import json

from llm_wiki_vs_rag.config import AppConfig
from llm_wiki_vs_rag.eval.harness import run_queries_for_system
from llm_wiki_vs_rag.eval.models import EvalQueryCase
from llm_wiki_vs_rag.models import GenerationResult
from llm_wiki_vs_rag.paths import ProjectPaths


def test_phase_targeted_query_run_fails_on_mixed_phase_input(tmp_path):
    paths = ProjectPaths(project_root=tmp_path)
    paths.ensure()
    (paths.artifacts_dir / "rag_index").mkdir(parents=True, exist_ok=True)
    (paths.artifacts_dir / "rag_index" / "manifest.json").write_text(
        json.dumps({"snapshot_id": "sha256:phase-2-snapshot", "execution_fingerprint": "sha256:exec-rag"}),
        encoding="utf-8",
    )

    try:
        run_queries_for_system(
            config=AppConfig(project_root=tmp_path),
            paths=paths,
            query_cases=[
                EvalQueryCase(query_id="q1", question="Q1", category="policy", phase="phase_1"),
                EvalQueryCase(query_id="q2", question="Q2", category="history", phase="phase_2"),
            ],
            system="rag",
            target_phase="phase_2",
        )
    except ValueError as exc:
        assert "Phase-targeted benchmark execution requires all query rows to match the requested phase" in str(exc)
        assert "q1" in str(exc)
        assert "phase_1" in str(exc)
    else:
        raise AssertionError("Expected explicit phase binding with mixed-phase input to fail.")


def test_phase_targeted_query_run_succeeds_when_all_rows_match(monkeypatch, tmp_path):
    artifact_dir = tmp_path / "artifacts" / "rag_runs" / "run-1"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "metadata.json").write_text(
        json.dumps({"mode": "rag", "corpus_snapshot": "sha256:phase-2-snapshot", "execution_fingerprint": "sha256:exec-rag"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "llm_wiki_vs_rag.eval.harness.run_rag_queries",
        lambda **_kwargs: [
            GenerationResult(query_id="q2::phase=phase_2", answer="a2", mode="rag", artifact_dir=str(artifact_dir)),
        ],
    )

    paths = ProjectPaths(project_root=tmp_path)
    paths.ensure()
    (paths.artifacts_dir / "rag_index").mkdir(parents=True, exist_ok=True)
    (paths.artifacts_dir / "rag_index" / "manifest.json").write_text(
        json.dumps({"snapshot_id": "sha256:phase-2-snapshot", "execution_fingerprint": "sha256:exec-rag"}),
        encoding="utf-8",
    )

    records = run_queries_for_system(
        config=AppConfig(project_root=tmp_path),
        paths=paths,
        query_cases=[
            EvalQueryCase(query_id="q2", question="Q2", category="history", phase="phase_2"),
        ],
        system="rag",
        target_phase="phase_2",
    )

    assert len(records) == 1
    assert records[0].phase == "phase_2"
    assert records[0].metadata.get("corpus_snapshot") == "sha256:phase-2-snapshot"
