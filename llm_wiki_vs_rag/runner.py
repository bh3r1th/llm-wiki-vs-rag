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
    if config.llm.mock_mode:
        raise ValueError("Benchmark commands cannot run with llm mock_mode enabled.")
    if provider != "openai-compatible":
        raise ValueError(f"Unsupported benchmark LLM provider: {config.llm.provider}.")
    if not config.llm.base_url:
        raise ValueError("Missing LLM configuration for benchmark commands: llm.base_url is required.")
    if not config.llm.api_key:
        raise ValueError("Missing LLM configuration for benchmark commands: llm.api_key is required.")


def _validate_comparison_cohorts(rag_outputs, wiki_outputs) -> None:
    rag_pairs = {(record.query_id, record.phase) for record in rag_outputs}
    wiki_pairs = {(record.query_id, record.phase) for record in wiki_outputs}
    if rag_pairs != wiki_pairs:
        rag_only = sorted(rag_pairs - wiki_pairs)[:5]
        wiki_only = sorted(wiki_pairs - rag_pairs)[:5]
        raise ValueError(
            "Cannot compare systems with mismatched (query_id, phase) cohorts. "
            f"rag_only_sample={rag_only}, wiki_only_sample={wiki_only}."
        )


def _validate_comparison_queryset_equivalence(rag_outputs, wiki_outputs) -> None:
    rag_by_pair = {(record.query_id, record.phase): record for record in rag_outputs}
    wiki_by_pair = {(record.query_id, record.phase): record for record in wiki_outputs}
    mismatches: list[dict[str, str]] = []
    for pair in sorted(rag_by_pair):
        rag_record = rag_by_pair[pair]
        wiki_record = wiki_by_pair[pair]
        if rag_record.question != wiki_record.question or rag_record.category != wiki_record.category:
            mismatches.append(
                {
                    "query_id": rag_record.query_id,
                    "phase": rag_record.phase,
                    "rag_question": rag_record.question,
                    "wiki_question": wiki_record.question,
                    "rag_category": rag_record.category,
                    "wiki_category": wiki_record.category,
                }
            )
    if mismatches:
        raise ValueError(
            "Cannot compare systems when matched (query_id, phase) rows differ in question/category "
            f"content. mismatch_sample={mismatches[:5]}."
        )


def _validate_system_uniqueness(outputs, system_name: str) -> None:
    counts: dict[tuple[str, str], int] = {}
    for record in outputs:
        key = (record.query_id, record.phase)
        counts[key] = counts.get(key, 0) + 1
    duplicates = sorted((query_id, phase, count) for (query_id, phase), count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(
            f"Cannot compare systems when one system has duplicate (query_id, phase) rows: "
            f"system={system_name}, duplicate_sample={duplicates[:5]}."
        )


def _validate_phase_snapshot_integrity(outputs, context: str) -> None:
    phase_rows = [record for record in outputs if record.phase in {"phase_1", "phase_2"}]
    phases_present = {record.phase for record in phase_rows}
    if not {"phase_1", "phase_2"}.issubset(phases_present):
        return

    by_phase: dict[str, set[str]] = {"phase_1": set(), "phase_2": set()}
    missing_snapshot_rows: list[tuple[str, str]] = []
    for record in phase_rows:
        snapshot = str(record.metadata.get("corpus_snapshot", "")).strip()
        if not snapshot:
            missing_snapshot_rows.append((record.query_id, record.phase))
            continue
        by_phase[record.phase].add(snapshot)

    if missing_snapshot_rows:
        raise ValueError(
            f"Missing corpus snapshot identity for phase comparison in {context}. "
            f"missing_sample={missing_snapshot_rows[:5]}."
        )
    if len(by_phase["phase_1"]) != 1:
        raise ValueError(
            f"Inconsistent phase_1 corpus snapshot mapping in {context}: "
            f"snapshots={sorted(by_phase['phase_1'])}."
        )
    if len(by_phase["phase_2"]) != 1:
        raise ValueError(
            f"Inconsistent phase_2 corpus snapshot mapping in {context}: "
            f"snapshots={sorted(by_phase['phase_2'])}."
        )
    if by_phase["phase_1"] == by_phase["phase_2"]:
        snapshot = next(iter(by_phase["phase_1"]))
        raise ValueError(
            "Phase comparison requires distinct corpus snapshots for phase_1 and phase_2 in "
            f"{context}, but both phases map to snapshot={snapshot}."
        )


def _validate_cross_system_phase_snapshot_parity(rag_outputs, wiki_outputs) -> None:
    for phase in ("phase_1", "phase_2"):
        rag_snapshots = {
            str(record.metadata.get("corpus_snapshot", "")).strip()
            for record in rag_outputs
            if record.phase == phase
        }
        wiki_snapshots = {
            str(record.metadata.get("corpus_snapshot", "")).strip()
            for record in wiki_outputs
            if record.phase == phase
        }
        rag_snapshots.discard("")
        wiki_snapshots.discard("")
        if not rag_snapshots and not wiki_snapshots:
            continue
        if rag_snapshots != wiki_snapshots:
            raise ValueError(
                f"Cross-system snapshot parity failed for {phase}: "
                f"rag_snapshots={sorted(rag_snapshots)}, wiki_snapshots={sorted(wiki_snapshots)}."
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
        _validate_phase_snapshot_integrity(outputs, context=f"{command}:{run_file}")
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
        _validate_phase_snapshot_integrity(rag_outputs, context=f"compare-systems rag:{rag_run_file}")
        _validate_phase_snapshot_integrity(wiki_outputs, context=f"compare-systems wiki:{wiki_run_file}")
        _validate_system_uniqueness(rag_outputs, "rag")
        _validate_system_uniqueness(wiki_outputs, "wiki")
        _validate_cross_system_phase_snapshot_parity(rag_outputs, wiki_outputs)
        _validate_comparison_cohorts(rag_outputs, wiki_outputs)
        _validate_comparison_queryset_equivalence(rag_outputs, wiki_outputs)
        outputs = rag_outputs + wiki_outputs
        labels = load_manual_labels(labels_file)
        records = merge_outputs_with_labels(outputs, labels)
        write_reports(records=records, output_dir=output_dir)
    else:
        raise ValueError(f"Unsupported command: {command}")
