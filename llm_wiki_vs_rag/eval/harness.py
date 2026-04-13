"""Evaluation harness entry points."""

from llm_wiki_vs_rag.config import AppConfig
from llm_wiki_vs_rag.eval.metrics import score_exact_match
from llm_wiki_vs_rag.eval.report import write_eval_report
from llm_wiki_vs_rag.models import EvalRecord
from llm_wiki_vs_rag.paths import ProjectPaths


def evaluate_queries(config: AppConfig, paths: ProjectPaths) -> list[EvalRecord]:
    """Run a minimal evaluation pass and persist report."""
    _ = config
    records = [
        EvalRecord(query_id="sample", mode="rag", score=score_exact_match("a", "a")),
        EvalRecord(query_id="sample", mode="wiki", score=score_exact_match("a", "b")),
    ]
    write_eval_report(records=records, output_path=paths.artifacts_dir / config.eval.output_name)
    return records
