# Re-audit (2026-04-13)

## Remaining critical benchmark credibility issues

1. **Execution fingerprint is not enforced against live query-time config.**
   Query paths load `execution_fingerprint` from manifests and copy it into per-query artifacts, but they do not verify that the current runtime config still hashes to that fingerprint before running retrieval and generation. This allows benchmark-significant drift (for example changing `rag.top_k`) while artifacts continue to claim the old execution identity.

2. **Phase-scoped query runs can silently drop non-target rows, enabling cherry-picked cohorts.**
   `run_queries_for_system(..., target_phase=...)` filters to matching phase rows and proceeds without requiring that all provided rows match the requested phase. That means a mixed query file can produce a partial subset run without an explicit failure, which can undermine fairness if this is used as a benchmark result.

## Remaining contract deviations

1. **No additional hard contract deviations found in the current code/tests beyond the credibility gaps above.**

## Dead code still worth removing before experiment

1. **None obvious in core benchmark runtime paths.**
