# Re-audit (2026-04-13)

## Remaining critical benchmark credibility issues

1. **Execution fingerprint is still not validated against live query-time runtime config.**
   Query execution reads `execution_fingerprint` from canonical manifests and propagates it into per-query metadata, but query paths do not recompute and compare the fingerprint from the current runtime config before generation. This still permits benchmark-significant config drift (for example `rag.top_k`, chunking params, or model ID changes) while outputs continue to claim the stale fingerprint.

2. **Phase-targeted query runs can still silently drop non-matching rows.**
   `run_queries_for_system(..., target_phase=...)` filters to matching rows and proceeds even when the supplied query file contains additional rows from other phases. That behavior still permits accidental or intentional cohort slicing without a hard failure.

## Remaining contract deviations

1. **No additional hard contract deviations identified beyond the two benchmark-credibility blockers above.**

## Dead code still worth removing before experiment

1. **None clearly blocking the experiment in core runtime paths.**
