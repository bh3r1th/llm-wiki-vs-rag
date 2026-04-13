# Re-audit (2026-04-13)

## Remaining critical benchmark credibility issues

1. **Execution fingerprint can misreport the actual model used.**
   `compute_execution_fingerprint()` hashes `config.llm.model_name`, while runtime model resolution in `LLMClient` can come from `LLM_MODEL` env fallback. If config keeps default `dummy-model` but env sets a real model, artifacts still claim the wrong model identity.

## Remaining contract deviations

1. **README run examples are not executable as written.**
   CLI requires `--phase` for `run-rag-queries` and `run-wiki-queries`, but README examples omit it.

## Dead code still worth removing before experiment

1. **None found in runtime benchmark paths worth immediate removal.**
