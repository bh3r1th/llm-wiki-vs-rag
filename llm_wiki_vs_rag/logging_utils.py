"""Logging setup helpers."""

import logging


def configure_logging(level: int = logging.INFO) -> None:
    """Configure a minimal global logging format for CLI runs."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
