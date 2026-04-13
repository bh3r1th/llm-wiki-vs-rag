"""Command line interface for llm_wiki_vs_rag."""

import argparse

from llm_wiki_vs_rag.config import AppConfig
from llm_wiki_vs_rag.logging_utils import configure_logging
from llm_wiki_vs_rag.runner import run_command


SUPPORTED_COMMANDS = ("build-rag-index", "wiki-ingest", "run-queries", "evaluate")


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(description="llm-wiki-vs-rag benchmark runner")
    parser.add_argument("command", choices=SUPPORTED_COMMANDS, help="Command to execute")
    return parser


def main() -> None:
    """CLI entry point."""
    configure_logging()
    parser = build_parser()
    args = parser.parse_args()
    config = AppConfig()
    run_command(command=args.command, config=config)


if __name__ == "__main__":
    main()
