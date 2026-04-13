from __future__ import annotations

import csv
import json

from llm_wiki_vs_rag.cli.main import build_parser
from llm_wiki_vs_rag.config import AppConfig
from llm_wiki_vs_rag.runner import run_command


def _write_run_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_make_review_pack_writes_csv_and_markdown(tmp_path):
    run_file = tmp_path / "run.jsonl"
    output_dir = tmp_path / "review"
    _write_run_jsonl(
        run_file,
        [
            {
                "system": "rag",
                "query_id": "q1",
                "phase": "phase_1",
                "category": "lookup",
                "question": "Question 1",
                "answer": "Answer 1",
                "metadata": {
                    "artifact_path": "artifacts/row1.json",
                    "corpus_snapshot": "snapshot-A",
                    "execution_fingerprint": "fp-A",
                },
            }
        ],
    )

    run_command(
        "make-review-pack",
        AppConfig(project_root=tmp_path),
        run_file=str(run_file),
        output_dir=str(output_dir),
    )

    assert (output_dir / "review_pack.csv").exists()
    assert (output_dir / "review_pack.md").exists()


def test_make_review_pack_groups_markdown_by_phase_then_category(tmp_path):
    run_file = tmp_path / "run.jsonl"
    output_dir = tmp_path / "review"
    _write_run_jsonl(
        run_file,
        [
            {
                "system": "wiki",
                "query_id": "q2",
                "phase": "phase_2",
                "category": "synthesis",
                "question": "Question 2",
                "answer": "Answer 2",
                "metadata": {"corpus_snapshot": "snapshot-B", "execution_fingerprint": "fp-B"},
            },
            {
                "system": "rag",
                "query_id": "q1",
                "phase": "phase_1",
                "category": "lookup",
                "question": "Question 1",
                "answer": "Answer 1",
                "metadata": {"corpus_snapshot": "snapshot-A", "execution_fingerprint": "fp-A"},
            },
        ],
    )

    run_command(
        "make-review-pack",
        AppConfig(project_root=tmp_path),
        run_file=str(run_file),
        output_dir=str(output_dir),
    )

    markdown = (output_dir / "review_pack.md").read_text(encoding="utf-8")
    assert "## phase_1" in markdown
    assert "### lookup" in markdown
    assert "## phase_2" in markdown
    assert "### synthesis" in markdown



def test_make_review_pack_rejects_duplicate_identity_rows(tmp_path):
    run_file = tmp_path / "run.jsonl"
    output_dir = tmp_path / "review"
    duplicate = {
        "system": "rag",
        "query_id": "q1",
        "phase": "phase_1",
        "category": "lookup",
        "question": "Question 1",
        "answer": "Answer 1",
        "metadata": {"corpus_snapshot": "snapshot-A", "execution_fingerprint": "fp-A"},
    }
    _write_run_jsonl(run_file, [duplicate, duplicate])

    try:
        run_command(
            "make-review-pack",
            AppConfig(project_root=tmp_path),
            run_file=str(run_file),
            output_dir=str(output_dir),
        )
    except ValueError as exc:
        assert "unique run output rows per (system, query_id, phase)" in str(exc)
    else:
        raise AssertionError("Expected duplicate identity rows to fail review pack generation.")


def test_make_review_pack_rejects_missing_required_metadata(tmp_path):
    run_file = tmp_path / "run.jsonl"
    output_dir = tmp_path / "review"
    _write_run_jsonl(
        run_file,
        [
            {
                "system": "wiki",
                "query_id": "q1",
                "phase": "phase_1",
                "category": "lookup",
                "question": "Question 1",
                "answer": "Answer 1",
                "metadata": {"corpus_snapshot": "snapshot-A"},
            }
        ],
    )

    try:
        run_command(
            "make-review-pack",
            AppConfig(project_root=tmp_path),
            run_file=str(run_file),
            output_dir=str(output_dir),
        )
    except ValueError as exc:
        assert "metadata.corpus_snapshot" in str(exc)
        assert "metadata.execution_fingerprint" in str(exc)
    else:
        raise AssertionError("Expected missing required metadata to fail review pack generation.")



def test_make_review_pack_rejects_missing_question_or_answer(tmp_path):
    run_file = tmp_path / "run.jsonl"
    output_dir = tmp_path / "review"
    _write_run_jsonl(
        run_file,
        [
            {
                "system": "wiki",
                "query_id": "q1",
                "phase": "phase_1",
                "category": "lookup",
                "question": "Question 1",
                "answer": "",
                "metadata": {
                    "corpus_snapshot": "snapshot-A",
                    "execution_fingerprint": "fp-A",
                },
            }
        ],
    )

    try:
        run_command(
            "make-review-pack",
            AppConfig(project_root=tmp_path),
            run_file=str(run_file),
            output_dir=str(output_dir),
        )
    except ValueError as exc:
        assert "non-empty question and answer" in str(exc)
    else:
        raise AssertionError("Expected missing question/answer to fail review pack generation.")



def test_make_review_pack_csv_contains_expected_columns(tmp_path):
    run_file = tmp_path / "run.jsonl"
    output_dir = tmp_path / "review"
    _write_run_jsonl(
        run_file,
        [
            {
                "system": "rag",
                "query_id": "q1",
                "phase": "phase_1",
                "category": "lookup",
                "question": "Question 1",
                "answer": "Answer 1",
                "metadata": {
                    "artifact_dir": "artifacts/row1",
                    "corpus_snapshot": "snapshot-A",
                    "execution_fingerprint": "fp-A",
                },
            }
        ],
    )

    run_command(
        "make-review-pack",
        AppConfig(project_root=tmp_path),
        run_file=str(run_file),
        output_dir=str(output_dir),
    )

    with (output_dir / "review_pack.csv").open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == [
            "system",
            "query_id",
            "phase",
            "category",
            "question",
            "answer",
            "artifact_path",
            "corpus_snapshot",
            "execution_fingerprint",
        ]



def test_cli_parser_supports_make_review_pack(tmp_path):
    parser = build_parser()
    args = parser.parse_args(
        [
            "make-review-pack",
            "--run-file",
            str(tmp_path / "run.jsonl"),
            "--output-dir",
            str(tmp_path / "review"),
        ]
    )
    assert args.command == "make-review-pack"
