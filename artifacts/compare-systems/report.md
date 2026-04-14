# RAG vs Wiki Evaluation Report

## System Summary

| System | N | Labeled | Accuracy % | Latest-State % | Contradiction Resolved % | Avg Latency (ms) | Avg Tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| rag | 48 | 48 | 12.5 | 87.5 | N/A | 120.374 | 2.0 |
| wiki | 48 | 48 | 100.0 | 100.0 | N/A | 2.215 | 2.0 |

## Drift (Phase 2 - Phase 1)

| System | Category | Accuracy Δ | Latest-State Δ | Contradiction-Resolved Δ |
|---|---|---:|---:|---:|
| rag | contradiction | 0.0 | 0.0 | - |
| rag | latest_state | -1.0 | -1.0 | - |
| rag | lookup | 0.0 | 0.0 | - |
| rag | synthesis | 0.0 | 0.0 | - |
| wiki | contradiction | 0.0 | 0.0 | - |
| wiki | latest_state | 0.0 | 0.0 | - |
| wiki | lookup | 0.0 | 0.0 | - |
| wiki | synthesis | 0.0 | 0.0 | - |
