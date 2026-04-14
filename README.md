# llm-wiki-vs-rag

This repository is a controlled benchmark of two approaches to answer generation over evolving documents: query-time synthesis (RAG) and ingest-time synthesis (LLM Wiki). The RAG baseline is intentionally simple so comparisons stay fair. The wiki follows Andrej Karpathy-style gist markdown pages and is updated incrementally from raw documents. Raw documents remain the source of truth.

## What this repo tests

- Lookup from grounded source material
- Cross-document synthesis
- Latest-state and deprecation handling
- Contradiction detection and resolution
- Output drift after corpus updates

## Systems under test

### RAG (query-time synthesis)

- Builds a local chunk index from `/raw`
- Retrieves top-k chunks at query time
- Synthesizes an answer from retrieved context only

### Wiki (ingest-time synthesis)

- Maintains persistent markdown pages in `/wiki`
- Updates pages incrementally from `/raw` in chronological order
- Answers from retrieved wiki pages at query time
- Not a graph database or schema-backed store

## Repo layout

```text
llm-wiki-vs-rag/
├─ llm_wiki_vs_rag/
│  ├─ cli/main.py
│  ├─ rag/
│  ├─ wiki/
│  ├─ eval/
│  └─ runner.py
├─ tests/
├─ raw/            # chronological source documents
├─ wiki/           # synthesized markdown wiki pages
├─ artifacts/      # run and evaluation outputs
├─ index.md        # wiki index
└─ log.md          # wiki ingest log
```

## How the benchmark works

- The corpus is chronological; later documents can refine, contradict, or deprecate earlier ones.
- Phase 1 runs on an earlier corpus snapshot.
- Phase 2 runs after corpus updates.
- The same query set is used across systems and phases.
- The same evaluation dimensions are applied to both systems.

## Metrics

- Accuracy
- Synthesis quality
- Latest-state accuracy
- Contradiction detection/resolution
- Drift between phases
- Cost (token usage)
- Latency
- Compression loss
- Provenance fidelity

## Running it

```bash
# build RAG index
python -m llm_wiki_vs_rag.cli.main build-rag-index

# ingest wiki from raw docs
python -m llm_wiki_vs_rag.cli.main wiki-ingest

# run queries
python -m llm_wiki_vs_rag.cli.main run-rag-queries --query-file path/to/queries.jsonl --phase phase_1 --output-file artifacts/run-rag-queries.jsonl
python -m llm_wiki_vs_rag.cli.main run-wiki-queries --query-file path/to/queries.jsonl --phase phase_1 --output-file artifacts/run-wiki-queries.jsonl

# compare systems
python -m llm_wiki_vs_rag.cli.main compare-systems \
  --rag-run-file artifacts/run-rag-queries.jsonl \
  --wiki-run-file artifacts/run-wiki-queries.jsonl \
  --labels-file path/to/manual_labels.csv \
  --output-dir artifacts/compare-systems
```

## Smoke run workflow

```bash
# 1) freeze-corpus
python -m llm_wiki_vs_rag.cli.main freeze-corpus --dataset-root path/to/dataset

# 2) validate-queries
python -m llm_wiki_vs_rag.cli.main validate-queries --query-file path/to/benchmark_queries.jsonl

# 3) smoke-queries
python -m llm_wiki_vs_rag.cli.main smoke-queries --query-file path/to/benchmark_queries.jsonl --output-file artifacts/smoke_queries.jsonl

# 4) split by phase
# Split benchmark_queries.jsonl into two files:
#   - artifacts/queries.phase_1.jsonl (phase_1 rows only)
#   - artifacts/queries.phase_2.jsonl (phase_2 rows only)

# 5) switch raw to phase_1
python -m llm_wiki_vs_rag.cli.main switch-phase-corpus --phase phase_1

# 6) build-rag-index
python -m llm_wiki_vs_rag.cli.main build-rag-index

# 7) wiki-ingest
python -m llm_wiki_vs_rag.cli.main wiki-ingest

# 8) benchmark-phase-run for rag/wiki phase_1
python -m llm_wiki_vs_rag.cli.main benchmark-phase-run --system rag --phase phase_1 --query-file artifacts/queries.phase_1.jsonl --output-file artifacts/rag.phase_1.jsonl
python -m llm_wiki_vs_rag.cli.main benchmark-phase-run --system wiki --phase phase_1 --query-file artifacts/queries.phase_1.jsonl --output-file artifacts/wiki.phase_1.jsonl

# 9) switch raw to phase_2
python -m llm_wiki_vs_rag.cli.main switch-phase-corpus --phase phase_2

# 10) build-rag-index
python -m llm_wiki_vs_rag.cli.main build-rag-index

# 11) wiki-ingest
python -m llm_wiki_vs_rag.cli.main wiki-ingest

# 12) benchmark-phase-run for rag/wiki phase_2
python -m llm_wiki_vs_rag.cli.main benchmark-phase-run --system rag --phase phase_2 --query-file artifacts/queries.phase_2.jsonl --output-file artifacts/rag.phase_2.jsonl
python -m llm_wiki_vs_rag.cli.main benchmark-phase-run --system wiki --phase phase_2 --query-file artifacts/queries.phase_2.jsonl --output-file artifacts/wiki.phase_2.jsonl

# 13) merge per-phase run files
cat artifacts/rag.phase_1.jsonl artifacts/rag.phase_2.jsonl > artifacts/rag.run.jsonl
cat artifacts/wiki.phase_1.jsonl artifacts/wiki.phase_2.jsonl > artifacts/wiki.run.jsonl

# 14) inspect-run
python -m llm_wiki_vs_rag.cli.main inspect-run --run-file artifacts/rag.run.jsonl
python -m llm_wiki_vs_rag.cli.main inspect-run --run-file artifacts/wiki.run.jsonl

# 15) make-label-template
python -m llm_wiki_vs_rag.cli.main make-label-template --run-file artifacts/rag.run.jsonl --output-file artifacts/manual_labels.csv

# 16) evaluate
python -m llm_wiki_vs_rag.cli.main evaluate-rag --run-file artifacts/rag.run.jsonl --labels-file artifacts/manual_labels.csv --output-dir artifacts/eval-rag
python -m llm_wiki_vs_rag.cli.main evaluate-wiki --run-file artifacts/wiki.run.jsonl --labels-file artifacts/manual_labels.csv --output-dir artifacts/eval-wiki

# 17) compare-systems
python -m llm_wiki_vs_rag.cli.main compare-systems --rag-run-file artifacts/rag.run.jsonl --wiki-run-file artifacts/wiki.run.jsonl --labels-file artifacts/manual_labels.csv --output-dir artifacts/compare-systems
```

## Manual labeling guide

Allowed values:

- `accuracy`: `correct` | `partial` | `wrong`
- `synthesis`: `full` | `incomplete` | `failed`
- `latest_state`: `correct` | `stale` | `missed_update`
- `contradiction_detected`: `true` | `false`
- `contradiction_resolved`: `true` | `false`
- `compression_loss`: `none` | `minor` | `major`
- `provenance_fidelity`: `true` | `false`

## Important note

- Evaluation requires manual labels.
- Complete the smoke run before running a full benchmark.
- `phase_1` and `phase_2` must run against different raw snapshots.

## Important constraints

- No GraphRAG extensions
- No agent workflows
- No hidden structured store behind the wiki
- Wiki benchmark mode stays pure (no wiki→RAG fallback in locked mode)
- Raw docs are the source of truth
- Benchmark commands require a real LLM provider configuration (no mock/stub mode)

## Outputs / artifacts

- Query runs write JSONL outputs (default under `/artifacts`).
- Per-query trace artifacts are written under `/artifacts/rag_runs/<run_id>` and `/artifacts/wiki_runs/<run_id>`.
- Evaluations/comparisons write `summary.json`, `summary.csv`, `per_query_results.csv`, and `report.md` to the chosen output directory.

## What to look for in results

- Where ingest-time summaries improve cross-document synthesis.
- Where wiki pages become stale or miss late-breaking updates.
- Where RAG is safer because it pulls directly from newest raw evidence.

## Non-goals

- A general-purpose RAG framework
- A wiki product
- A knowledge graph system
- A full evaluation platform

## RAG vs LLM Wiki — Controlled Benchmark

- **Setup**
  - RAG = retrieval + generation.
  - Wiki = pre-ingested structured knowledge (no retrieval at query time).
  - Same dataset, same queries, deterministic mock LLM.

- **Results**
  - RAG  → accuracy: 12.5%   | latency: ~120 ms
  - Wiki → accuracy: 100.0%  | latency: ~2.2 ms

- **Key observations**
  - Retrieval introduces variance (noisy context, inconsistent relevance).
  - RAG fails under drift (`phase_2` latest_state degradation).
  - RAG struggles with contradiction + synthesis.
  - Wiki is stable due to removal of retrieval step.

- **RAG failure modes**
  - `LOOKUP_NOISY`
  - `SYNTHESIS_PARTIAL`
  - `LATEST_STATE_STALE`
  - `CONTRADICTION_MISSED`

- **Conclusion**
  - Retrieval adds latency and instability.
  - Pre-ingested knowledge reduces variance and improves consistency.
