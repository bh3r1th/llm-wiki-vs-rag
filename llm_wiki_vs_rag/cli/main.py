"""Command line interface for llm_wiki_vs_rag."""

from __future__ import annotations

import argparse
from pathlib import Path

from llm_wiki_vs_rag.config import AppConfig
from llm_wiki_vs_rag.logging_utils import configure_logging
from llm_wiki_vs_rag.runner import run_command

def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(description="llm-wiki-vs-rag benchmark runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("build-rag-index")
    subparsers.add_parser("wiki-ingest")

    for run_command_name in ("run-rag-queries", "run-wiki-queries"):
        run_parser = subparsers.add_parser(run_command_name)
        run_parser.add_argument("--query-file", type=Path, required=True)
        run_parser.add_argument("--phase", choices=("phase_1", "phase_2"), required=True)
        run_parser.add_argument("--output-file", type=Path)

    validate_queries_parser = subparsers.add_parser("validate-queries")
    validate_queries_parser.add_argument("--query-file", type=Path, required=True)

    smoke_parser = subparsers.add_parser("smoke-queries")
    smoke_parser.add_argument("--query-file", type=Path, required=True)
    smoke_parser.add_argument("--output-file", type=Path)

    phase_run_parser = subparsers.add_parser("benchmark-phase-run")
    phase_run_parser.add_argument("--system", choices=("rag", "wiki"), required=True)
    phase_run_parser.add_argument("--phase", choices=("phase_1", "phase_2"), required=True)
    phase_run_parser.add_argument("--query-file", type=Path, required=True)
    phase_run_parser.add_argument("--output-file", type=Path)

    for eval_command_name in ("evaluate-rag", "evaluate-wiki"):
        eval_parser = subparsers.add_parser(eval_command_name)
        eval_parser.add_argument("--run-file", type=Path, required=True)
        eval_parser.add_argument("--labels-file", type=Path, required=True)
        eval_parser.add_argument("--output-dir", type=Path)

    freeze_parser = subparsers.add_parser("freeze-corpus")
    freeze_parser.add_argument("--dataset-root", type=Path, required=True)

    switch_parser = subparsers.add_parser("switch-phase-corpus")
    switch_parser.add_argument("--phase", choices=("phase_1", "phase_2"), required=True)
    switch_parser.add_argument("--source-root", type=Path)

    compare_parser = subparsers.add_parser("compare-systems")
    compare_parser.add_argument("--rag-run-file", type=Path, required=True)
    compare_parser.add_argument("--wiki-run-file", type=Path, required=True)
    compare_parser.add_argument("--labels-file", type=Path, required=True)
    compare_parser.add_argument("--output-dir", type=Path)

    label_template_parser = subparsers.add_parser("make-label-template")
    label_template_parser.add_argument("--run-file", type=Path, required=True)
    label_template_parser.add_argument("--output-file", type=Path, required=True)

    inspect_run_parser = subparsers.add_parser("inspect-run")
    inspect_run_parser.add_argument("--run-file", type=Path, required=True)

    return parser


def main() -> None:
    """CLI entry point."""
    configure_logging()
    parser = build_parser()
    args = parser.parse_args()
    config = AppConfig()

    kwargs = {
        key: value
        for key, value in vars(args).items()
        if key != "command" and value is not None
    }
    run_command(command=args.command, config=config, **kwargs)


if __name__ == "__main__":
    main()
