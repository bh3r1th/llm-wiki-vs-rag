# Re-audit (2026-04-13, post-fix)

## 1) Remaining critical benchmark credibility issues

- **Phase drift can still be computed on non-equivalent query sets across phases.**
  Current guards require both `phase_1` and `phase_2`, single-snapshot-per-phase, and chronology order, but do **not** require the same query identities/categories to appear in both phases within a system. This allows cherry-picked phase cohorts to pass validation and produce misleading drift deltas.

## 2) Remaining contract deviations

- **"Same query set is used across systems and phases" is only half-enforced.**
  Implementation enforces cross-system cohort parity on `(query_id, phase)` pairs, but there is no explicit intra-system phase parity check (`phase_1` query set == `phase_2` query set). This deviates from the stated benchmark contract.

- **"Benchmark commands require a real LLM provider configuration" is not uniformly enforced.**
  The explicit benchmark LLM config gate is applied to `wiki-ingest` and query-run commands, but not to other benchmark entrypoints (`build-rag-index`, `evaluate-*`, `compare-systems`).

## 3) Dead code still worth removing before experiment

- **`load_run_outputs()` JSON-array fallback path is likely unnecessary benchmark surface area.**
  The benchmark writer always emits JSONL (`save_run_outputs`), and CLI/docs contract run files as JSONL. Keeping a second format path adds parsing surface and contract ambiguity without clear experiment value.
