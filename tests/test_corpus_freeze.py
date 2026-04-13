from __future__ import annotations

import json

from llm_wiki_vs_rag.cli.main import build_parser
from llm_wiki_vs_rag.config import AppConfig
from llm_wiki_vs_rag.data.corpus_freeze import build_corpus_manifest, write_corpus_manifest
from llm_wiki_vs_rag.runner import run_command


def _write_phase_docs(root, phase: str, *, count: int, prefix_offset: int = 0, duplicate_name: str | None = None):
    phase_dir = root / phase
    phase_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(1, count + 1):
        number = idx + prefix_offset
        name = duplicate_name if duplicate_name and idx == count else f"{number:03d}_doc.txt"
        (phase_dir / name).write_text(f"{phase}-{idx}", encoding="utf-8")


def test_corpus_freeze_validates_phase_counts(tmp_path):
    _write_phase_docs(tmp_path, "phase_1", count=49)
    _write_phase_docs(tmp_path, "phase_2", count=50)

    try:
        build_corpus_manifest(tmp_path)
    except ValueError as exc:
        assert "exactly 50 files in phase_1" in str(exc)
    else:
        raise AssertionError("Expected phase count validation failure.")


def test_corpus_freeze_rejects_duplicate_doc_id(tmp_path):
    _write_phase_docs(tmp_path, "phase_1", count=50)
    _write_phase_docs(tmp_path, "phase_2", count=50)
    (tmp_path / "phase_2" / "001_doc.txt").write_text("phase_2 duplicate", encoding="utf-8")

    try:
        build_corpus_manifest(tmp_path)
    except ValueError as exc:
        assert "unique doc_id" in str(exc)
    else:
        raise AssertionError("Expected duplicate doc_id validation failure.")


def test_corpus_freeze_rejects_non_numbered_order(tmp_path):
    _write_phase_docs(tmp_path, "phase_1", count=50)
    _write_phase_docs(tmp_path, "phase_2", count=50)
    (tmp_path / "phase_2" / "abc_doc.txt").write_text("invalid", encoding="utf-8")
    (tmp_path / "phase_2" / "050_doc.txt").unlink()

    try:
        build_corpus_manifest(tmp_path)
    except ValueError as exc:
        assert "leading numeric prefix" in str(exc)
    else:
        raise AssertionError("Expected numbered filename validation failure.")


def test_corpus_freeze_writes_manifest_outputs(tmp_path):
    _write_phase_docs(tmp_path, "phase_1", count=50)
    _write_phase_docs(tmp_path, "phase_2", count=50, prefix_offset=50)

    output_dir = tmp_path / "data"
    manifest_path, summary_path = write_corpus_manifest(tmp_path, output_dir)

    assert manifest_path.exists()
    assert summary_path.exists()

    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert len(rows) == 100
    assert rows[0]["order_index"] == 1
    assert rows[-1]["order_index"] == 100
    assert summary["total_docs"] == 100
    assert summary["phase_1_docs"] == 50
    assert summary["phase_2_docs"] == 50


def test_cli_and_runner_support_freeze_corpus_command(tmp_path):
    _write_phase_docs(tmp_path / "dataset", "phase_1", count=50)
    _write_phase_docs(tmp_path / "dataset", "phase_2", count=50, prefix_offset=50)

    parser = build_parser()
    args = parser.parse_args(["freeze-corpus", "--dataset-root", str(tmp_path / "dataset")])
    assert args.command == "freeze-corpus"

    run_command("freeze-corpus", AppConfig(project_root=tmp_path), dataset_root=str(tmp_path / "dataset"))

    assert (tmp_path / "data" / "manifest.jsonl").exists()
    assert (tmp_path / "data" / "manifest_summary.json").exists()
