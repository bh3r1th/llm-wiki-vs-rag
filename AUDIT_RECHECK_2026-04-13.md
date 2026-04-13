# Re-audit (2026-04-13)

## Remaining critical benchmark credibility issues

1. **`evaluate-*` and `compare-systems` still do not enforce execution-fingerprint cohort integrity.**
   They validate `corpus_snapshot` and `corpus_order` consistency, but they do not require a single `execution_fingerprint` per `(system, phase)` cohort and do not require the field to exist on loaded run rows. A mixed run file can therefore still blend outputs from different benchmark configurations into one reported score if snapshot/order happen to match.

## Remaining contract deviations

1. **Report/evaluation contract still allows missing execution fingerprint in externally supplied run files.**
   Runtime query execution writes and validates `execution_fingerprint`, but `evaluate-rag`, `evaluate-wiki`, and `compare-systems` do not reject run rows where `metadata.execution_fingerprint` is absent or empty.

## Dead code still worth removing before experiment

1. **No high-risk dead code remains in the core benchmark/eval path.**
   No obviously orphaned logic in `runner.py`, query pipelines, or eval harness appears to threaten experiment integrity; cleanup can be deferred.
