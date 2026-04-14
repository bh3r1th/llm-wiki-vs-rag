# llm-wiki-vs-rag

Controlled benchmark comparing query-time synthesis (RAG) vs ingest-time synthesis (LLM Wiki) on an evolving document corpus.

## Scope

- Two systems: `rag` and `wiki`
- Two corpus phases: `phase_1` and `phase_2`
- Four query categories: `lookup`, `synthesis`, `latest_state`, `contradiction`
- Shared evaluation schema across both systems

## Benchmark modes

- **Controlled benchmark mode (documented by shipped artifacts):** deterministic/simulated behavior in the released run outputs under `artifacts/final_v6/`.
- **Provider-backed CLI mode (optional):** benchmark CLI commands validate provider configuration and reject `llm.mock_mode`.

These are distinct paths. The findings below come from the controlled artifacts, not from a live provider run.

## Reproducible artifacts (current repo state)

Use these files as the source of truth for the current benchmark snapshot:

- Runs:
  - `artifacts/final_v6/rag_phase_1_v6.jsonl`
  - `artifacts/final_v6/rag_phase_2_v6.jsonl`
  - `artifacts/final_v6/wiki_phase_1_v6.jsonl`
  - `artifacts/final_v6/wiki_phase_2_v6.jsonl`
  - `artifacts/final_v6/rag_full_runs_v6.jsonl`
  - `artifacts/final_v6/wiki_full_runs_v6.jsonl`
- Labels:
  - `artifacts/final_v6/full_labels_v6.csv`
- Summaries:
  - `artifacts/final_v6/summary.json`
  - `artifacts/final_v6/wiki_vs_rag_summary.json`

## What is supported by `artifacts/final_v6/summary.json`

- **Latency gap:**
  - RAG avg latency: **120.374 ms**
  - Wiki avg latency: **2.215 ms**
- **Controlled degradation behavior:**
  - RAG accuracy: **12.5%**
  - Wiki accuracy: **100.0%**
- **`latest_state` drift effect:**
  - RAG `latest_state` correctness: **87.5%** overall
  - Drift entry for RAG `latest_state` shows `latest_state_correct_rate_delta = -1.0` from phase 1 to phase 2
- **Synthetic/category-aware outputs:**
  - Category summaries are present for `lookup`, `synthesis`, `latest_state`, and `contradiction`

## Contradiction caveat (important)

- Native contradiction metric fields are zero in the final summary (`detected = 0`, `resolved = 0`, `resolved_pct = null`).
- Therefore this README does **not** claim measured contradiction detection/resolution success.
- Any contradiction failure behavior in these runs is from controlled/simulated output behavior (for example category-specific mock patterns), not from non-zero native contradiction metric success.

## Artifact layout

- Per-run traces: `artifacts/rag_runs/<run_id>/` and `artifacts/wiki_runs/<run_id>/`
- Comparison outputs (when generated): `summary.json`, `summary.csv`, `per_query_results.csv`, `report.md`

## Non-goals

- Production RAG framework
- Wiki product
- Knowledge graph system
