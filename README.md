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
