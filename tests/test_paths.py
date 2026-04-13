"""Tests for project path helper behavior."""

from llm_wiki_vs_rag.paths import ProjectPaths


def test_project_paths_ensure_creates_expected_layout(tmp_path):
    paths = ProjectPaths(project_root=tmp_path)

    paths.ensure()

    assert paths.raw_dir.exists()
    assert paths.wiki_dir.exists()
    assert paths.artifacts_dir.exists()
    assert paths.index_md.exists()
    assert paths.log_md.exists()
