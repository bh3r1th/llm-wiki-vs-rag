from __future__ import annotations

import csv
import json

from llm_wiki_vs_rag.cli.main import build_parser
from llm_wiki_vs_rag.config import AppConfig
from llm_wiki_vs_rag.runner import run_command


def _write_run_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_make_label_template_row_count_matches_run_output_count(tmp_path):
    run_file = tmp_path / "run.jsonl"
    output_file = tmp_path / "labels_template.csv"
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
                    "corpus_snapshot": "snapshot-A",
                    "execution_fingerprint": "fp-A",
                },
            },
            {
                "system": "rag",
                "query_id": "q2",
                "phase": "phase_2",
                "category": "synthesis",
                "question": "Question 2",
                "answer": "Answer 2",
                "metadata": {
                    "corpus_snapshot": "snapshot-B",
                    "execution_fingerprint": "fp-B",
                },
            },
        ],
    )

    run_command(
        "make-label-template",
        AppConfig(project_root=tmp_path),
        run_file=str(run_file),
        output_file=str(output_file),
    )

    with output_file.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2


def test_make_label_template_rejects_duplicate_identity_rows(tmp_path):
    run_file = tmp_path / "run.jsonl"
    output_file = tmp_path / "labels_template.csv"
    duplicate = {
        "system": "rag",
        "query_id": "q1",
        "phase": "phase_1",
        "category": "lookup",
        "question": "Question 1",
        "answer": "Answer 1",
        "metadata": {
            "corpus_snapshot": "snapshot-A",
            "execution_fingerprint": "fp-A",
        },
    }
    _write_run_jsonl(run_file, [duplicate, duplicate])

    try:
        run_command(
            "make-label-template",
            AppConfig(project_root=tmp_path),
            run_file=str(run_file),
            output_file=str(output_file),
        )
    except ValueError as exc:
        assert "unique run output rows per (system, query_id, phase)" in str(exc)
    else:
        raise AssertionError("Expected duplicate identity rows to fail template generation.")


def test_make_label_template_writes_all_required_columns(tmp_path):
    run_file = tmp_path / "run.jsonl"
    output_file = tmp_path / "labels_template.csv"
    _write_run_jsonl(
        run_file,
        [
            {
                "system": "wiki",
                "query_id": "q1",
                "phase": "phase_1",
                "category": "latest_state",
                "question": "Question",
                "answer": "Answer",
                "metadata": {
                    "corpus_snapshot": "snapshot-A",
                    "execution_fingerprint": "fp-A",
                },
            }
        ],
    )

    run_command(
        "make-label-template",
        AppConfig(project_root=tmp_path),
        run_file=str(run_file),
        output_file=str(output_file),
    )

    with output_file.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        expected_columns = [
            "system",
            "query_id",
            "phase",
            "category",
            "question",
            "answer",
            "corpus_snapshot",
            "execution_fingerprint",
            "accuracy",
            "synthesis",
            "latest_state",
            "contradiction_detected",
            "contradiction_resolved",
            "compression_loss",
            "provenance_fidelity",
            "evaluator_notes",
        ]
        assert reader.fieldnames == expected_columns


def test_make_label_template_fails_when_required_metadata_missing(tmp_path):
    run_file = tmp_path / "run.jsonl"
    output_file = tmp_path / "labels_template.csv"
    _write_run_jsonl(
        run_file,
        [
            {
                "system": "wiki",
                "query_id": "q1",
                "phase": "phase_1",
                "category": "latest_state",
                "question": "Question",
                "answer": "Answer",
                "metadata": {
                    "execution_fingerprint": "fp-A",
                },
            }
        ],
    )

    try:
        run_command(
            "make-label-template",
            AppConfig(project_root=tmp_path),
            run_file=str(run_file),
            output_file=str(output_file),
        )
    except ValueError as exc:
        assert "metadata.corpus_snapshot" in str(exc)
    else:
        raise AssertionError("Expected missing required metadata to fail template generation.")


def test_cli_parser_supports_make_label_template(tmp_path):
    parser = build_parser()
    args = parser.parse_args(
        [
            "make-label-template",
            "--run-file",
            str(tmp_path / "run.jsonl"),
            "--output-file",
            str(tmp_path / "labels.csv"),
        ]
    )
    assert args.command == "make-label-template"
