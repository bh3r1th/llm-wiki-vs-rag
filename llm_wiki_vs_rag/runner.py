"""Runner skeleton for benchmark command orchestration."""

from llm_wiki_vs_rag.config import AppConfig
from llm_wiki_vs_rag.eval.harness import evaluate_queries
from llm_wiki_vs_rag.models import QueryCase
from llm_wiki_vs_rag.paths import ProjectPaths
from llm_wiki_vs_rag.rag.pipeline import build_rag_index
from llm_wiki_vs_rag.rag.pipeline import run_rag_queries
from llm_wiki_vs_rag.wiki.pipeline import ingest_wiki
from llm_wiki_vs_rag.wiki.pipeline import run_wiki_queries


def run_command(command: str, config: AppConfig) -> None:
    """Dispatch supported runner commands to pipeline entry points."""
    paths = ProjectPaths(config.project_root)
    paths.ensure()

    if command == "build-rag-index":
        build_rag_index(config=config, paths=paths)
    elif command == "wiki-ingest":
        ingest_wiki(config=config, paths=paths)
    elif command == "run-queries":
        query_cases = [QueryCase(query_id="sample", question="What is the sample question?")]
        run_rag_queries(config=config, paths=paths, query_cases=query_cases)
        run_wiki_queries(config=config, paths=paths, query_cases=query_cases)
    elif command == "evaluate":
        evaluate_queries(config=config, paths=paths)
    else:
        raise ValueError(f"Unsupported command: {command}")
