"""Runner command orchestration."""

from __future__ import annotations

from pathlib import Path

from llm_wiki_vs_rag.config import AppConfig
from llm_wiki_vs_rag.eval.harness import (
    load_manual_labels,
    load_query_cases,
    load_run_outputs,
    merge_outputs_with_labels,
    run_queries_for_system,
    save_run_outputs,
)
from llm_wiki_vs_rag.eval.report import write_reports
from llm_wiki_vs_rag.paths import ProjectPaths
from llm_wiki_vs_rag.rag.pipeline import build_rag_index
from llm_wiki_vs_rag.wiki.pipeline import ingest_wiki


def _validate_benchmark_llm_config(config: AppConfig) -> None:
    provider = config.llm.provider.lower()
    if config.llm.mock_mode or provider in {"mock", "stub"}:
        raise ValueError("Benchmark commands cannot run with llm mock/stub mode enabled.")
    if provider != "openai-compatible":
        raise ValueError(f"Unsupported benchmark LLM provider: {config.llm.provider}.")
    if not config.llm.base_url:
        raise ValueError("Missing LLM configuration for benchmark commands: llm.base_url is required.")
    if not config.llm.api_key:
        raise ValueError("Missing LLM configuration for benchmark commands: llm.api_key is required.")
    if not config.llm.model_name:
        raise ValueError("Missing LLM configuration for benchmark commands: llm.model_name is required.")


def _validate_comparison_cohorts(rag_outputs, wiki_outputs) -> None:
    rag_query_ids = {record.query_id for record in rag_outputs}
    wiki_query_ids = {record.query_id for record in wiki_outputs}
    if rag_query_ids != wiki_query_ids:
        rag_only = sorted(rag_query_ids - wiki_query_ids)
        wiki_only = sorted(wiki_query_ids - rag_query_ids)
        raise ValueError(
            "Cannot compare systems with mismatched query_id cohorts. "
            f"rag_only={rag_only}, wiki_only={wiki_only}."
        )

    rag_phases = {record.phase for record in rag_outputs}
    wiki_phases = {record.phase for record in wiki_outputs}
    if rag_phases != wiki_phases:
        rag_only = sorted(rag_phases - wiki_phases)
        wiki_only = sorted(wiki_phases - rag_phases)
        raise ValueError(
            "Cannot compare systems with mismatched phase cohorts. "
            f"rag_only={rag_only}, wiki_only={wiki_only}."
        )


def run_command(command: str, config: AppConfig, **kwargs: str | None) -> None:
    """Dispatch supported runner commands to pipeline entry points."""
    paths = ProjectPaths(config.project_root)
    paths.ensure()

    if command == "build-rag-index":
        build_rag_index(config=config, paths=paths)
    elif command == "wiki-ingest":
        if config.benchmark.locked:
            _validate_benchmark_llm_config(config)
        ingest_wiki(config=config, paths=paths)
    elif command in {"run-rag-queries", "run-wiki-queries"}:
        _validate_benchmark_llm_config(config)
        query_file = Path(str(kwargs["query_file"]))
        output_file = Path(str(kwargs.get("output_file") or paths.artifacts_dir / f"{command}.jsonl"))
        query_cases = load_query_cases(query_file)
        system = "rag" if command == "run-rag-queries" else "wiki"
        save_run_outputs(run_queries_for_system(config=config, paths=paths, query_cases=query_cases, system=system), output_file)
    elif command in {"evaluate-rag", "evaluate-wiki"}:
        run_file = Path(str(kwargs["run_file"]))
        labels_file = Path(str(kwargs["labels_file"]))
        output_dir = Path(str(kwargs.get("output_dir") or paths.artifacts_dir / command))
        outputs = load_run_outputs(run_file)
        labels = load_manual_labels(labels_file)
        records = merge_outputs_with_labels(outputs, labels)
        write_reports(records=records, output_dir=output_dir)
    elif command == "compare-systems":
        rag_run_file = Path(str(kwargs["rag_run_file"]))
        wiki_run_file = Path(str(kwargs["wiki_run_file"]))
        labels_file = Path(str(kwargs["labels_file"]))
        output_dir = Path(str(kwargs.get("output_dir") or paths.artifacts_dir / "compare-systems"))
        rag_outputs = load_run_outputs(rag_run_file)
        wiki_outputs = load_run_outputs(wiki_run_file)
        _validate_comparison_cohorts(rag_outputs, wiki_outputs)
        outputs = rag_outputs + wiki_outputs
        labels = load_manual_labels(labels_file)
        records = merge_outputs_with_labels(outputs, labels)
        write_reports(records=records, output_dir=output_dir)
    else:
        raise ValueError(f"Unsupported command: {command}")
