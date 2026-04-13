# Re-audit (2026-04-13)

## Remaining critical benchmark credibility issues

1. **`evaluate-*` / `compare-systems` still accept mixed execution fingerprints within a system/phase cohort.**
   Runtime query execution now validates the manifest fingerprint, but downstream report paths only validate snapshot and chronology tokens. There is still no hard check that all rows for a given `(system, phase)` were produced under a single benchmark execution fingerprint, so mixed-config run files can still be merged into one score.

## Remaining contract deviations

1. **None newly identified beyond the fingerprint-cohort integrity gap above.**

## Dead code still worth removing before experiment

1. **No additional dead code stands out as pre-experiment risk in the core benchmark path.**
