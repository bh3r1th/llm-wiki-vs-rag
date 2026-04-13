"""Evaluation metric implementations."""

from __future__ import annotations

from collections import defaultdict

from llm_wiki_vs_rag.eval.models import DriftSummary, EvalSummary, EvaluationRecord


def _pct(count: int, total: int) -> float:
    return round((count / total) * 100.0, 2) if total else 0.0


def _avg(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 3) if values else None


def _validate_contradiction_invariants(records: list[EvaluationRecord]) -> None:
    offending = [
        {"system": record.system, "query_id": record.query_id, "phase": record.phase}
        for record in records
        if record.contradiction_resolved is True and record.contradiction_detected is not True
    ]
    if offending:
        raise ValueError(
            "Invalid contradiction labels in evaluation records: contradiction_resolved=True requires "
            "contradiction_detected=True. "
            f"offending_sample={offending[:5]}"
        )


def _validate_phase_cohort_alignment(records: list[EvaluationRecord]) -> None:
    grouped: dict[tuple[str, str], list[EvaluationRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.system, record.category)].append(record)

    mismatches: list[dict[str, object]] = []
    for (system, category), bucket in sorted(grouped.items()):
        phase_1 = {
            (record.query_id, record.question, record.category)
            for record in bucket
            if record.phase == "phase_1"
        }
        phase_2 = {
            (record.query_id, record.question, record.category)
            for record in bucket
            if record.phase == "phase_2"
        }
        if phase_1 == phase_2:
            continue
        only_phase_1 = sorted(phase_1 - phase_2)
        only_phase_2 = sorted(phase_2 - phase_1)
        mismatches.append(
            {
                "system": system,
                "category": category,
                "phase_1_only_sample": [
                    {"query_id": qid, "question": question, "category": cat}
                    for qid, question, cat in only_phase_1[:3]
                ],
                "phase_2_only_sample": [
                    {"query_id": qid, "question": question, "category": cat}
                    for qid, question, cat in only_phase_2[:3]
                ],
            }
        )
    if mismatches:
        raise ValueError(
            "Phase drift requires identical phase_1 and phase_2 query cohorts by (query_id, question, category). "
            f"mismatch_sample={mismatches[:5]}"
        )


def summarize_records(records: list[EvaluationRecord], group_fields: tuple[str, ...]) -> list[EvalSummary]:
    """Aggregate records by requested dimensions and compute percentages."""
    labeled_records = [record for record in records if record.is_labeled]
    _validate_contradiction_invariants(labeled_records)

    grouped: dict[tuple[str, ...], list[EvaluationRecord]] = defaultdict(list)
    for record in records:
        key = tuple(str(getattr(record, field)) for field in group_fields)
        grouped[key].append(record)

    summaries: list[EvalSummary] = []
    for key, bucket in sorted(grouped.items()):
        labeled = [record for record in bucket if record.is_labeled]
        latencies = [record.latency_ms for record in bucket if record.latency_ms is not None]
        total_tokens = [float(record.total_tokens) for record in bucket if record.total_tokens is not None]

        metrics: dict[str, dict[str, float | int | str | None]] = {}
        if labeled:
            contradiction_detected_total = sum(1 for record in labeled if record.contradiction_detected)
            contradiction_resolved_total = sum(1 for record in labeled if record.contradiction_resolved)
            if contradiction_resolved_total > contradiction_detected_total:
                raise ValueError(
                    "Invalid contradiction metrics input: resolved count exceeds detected count. "
                    f"group_by={dict(zip(group_fields, key))}, detected={contradiction_detected_total}, resolved={contradiction_resolved_total}"
                )
            accuracy_counts = {
                "correct": sum(1 for record in labeled if record.accuracy == "correct"),
                "partial": sum(1 for record in labeled if record.accuracy == "partial"),
                "wrong": sum(1 for record in labeled if record.accuracy == "wrong"),
            }
            latest_counts = {
                "correct": sum(1 for record in labeled if record.latest_state == "correct"),
                "stale": sum(1 for record in labeled if record.latest_state == "stale"),
                "missed_update": sum(1 for record in labeled if record.latest_state == "missed_update"),
            }
            metrics = {
                "accuracy": {
                    **accuracy_counts,
                    "correct_pct": _pct(accuracy_counts["correct"], len(labeled)),
                    "partial_pct": _pct(accuracy_counts["partial"], len(labeled)),
                    "wrong_pct": _pct(accuracy_counts["wrong"], len(labeled)),
                },
                "synthesis": {
                    "full": sum(1 for record in labeled if record.synthesis == "full"),
                    "incomplete": sum(1 for record in labeled if record.synthesis == "incomplete"),
                    "failed": sum(1 for record in labeled if record.synthesis == "failed"),
                },
                "latest_state": {
                    **latest_counts,
                    "correct_pct": _pct(latest_counts["correct"], len(labeled)),
                },
                "contradiction": {
                    "detected": contradiction_detected_total,
                    "resolved": contradiction_resolved_total,
                    "resolved_applicable_total": contradiction_detected_total,
                    "resolved_pct": (
                        _pct(contradiction_resolved_total, contradiction_detected_total)
                        if contradiction_detected_total
                        else None
                    ),
                },
                "compression_loss": {
                    "none": sum(1 for record in labeled if record.compression_loss == "none"),
                    "minor": sum(1 for record in labeled if record.compression_loss == "minor"),
                    "major": sum(1 for record in labeled if record.compression_loss == "major"),
                },
                "provenance_fidelity": {
                    "true": sum(1 for record in labeled if record.provenance_fidelity),
                    "false": sum(1 for record in labeled if record.provenance_fidelity is False),
                    "true_pct": _pct(
                        sum(1 for record in labeled if record.provenance_fidelity),
                        len(labeled),
                    ),
                },
            }

        summaries.append(
            EvalSummary(
                group_by=dict(zip(group_fields, key)),
                total=len(bucket),
                labeled_total=len(labeled),
                avg_latency_ms=_avg([float(value) for value in latencies]),
                avg_total_tokens=_avg(total_tokens),
                metrics=metrics,
            )
        )
    return summaries


def compute_drift(records: list[EvaluationRecord]) -> list[DriftSummary]:
    """Compute phase-2 minus phase-1 deltas for key quality indicators."""
    labeled_records = [record for record in records if record.is_labeled]
    _validate_contradiction_invariants(labeled_records)
    _validate_phase_cohort_alignment(labeled_records)

    grouped: dict[tuple[str, str], list[EvaluationRecord]] = defaultdict(list)
    for record in labeled_records:
        grouped[(record.system, record.category)].append(record)

    drifts: list[DriftSummary] = []
    for (system, category), bucket in sorted(grouped.items()):
        phase_1 = [record for record in bucket if record.phase == "phase_1"]
        phase_2 = [record for record in bucket if record.phase == "phase_2"]

        def rate(
            items: list[EvaluationRecord],
            attr: str,
            expected: str | bool,
            *,
            denominator_filter_attr: str | None = None,
            denominator_filter_value: str | bool | None = None,
        ) -> float | None:
            denominator_items = items
            if denominator_filter_attr is not None:
                denominator_items = [
                    item
                    for item in items
                    if getattr(item, denominator_filter_attr) == denominator_filter_value
                ]
            if not denominator_items:
                return None
            return round(sum(1 for item in denominator_items if getattr(item, attr) == expected) / len(denominator_items), 4)

        p1_accuracy = rate(phase_1, "accuracy", "correct")
        p2_accuracy = rate(phase_2, "accuracy", "correct")
        p1_latest = rate(phase_1, "latest_state", "correct")
        p2_latest = rate(phase_2, "latest_state", "correct")
        p1_resolved = rate(
            phase_1,
            "contradiction_resolved",
            True,
            denominator_filter_attr="contradiction_detected",
            denominator_filter_value=True,
        )
        p2_resolved = rate(
            phase_2,
            "contradiction_resolved",
            True,
            denominator_filter_attr="contradiction_detected",
            denominator_filter_value=True,
        )

        drifts.append(
            DriftSummary(
                system=system,
                category=category,
                phase_1_count=len(phase_1),
                phase_2_count=len(phase_2),
                accuracy_correct_rate_delta=None if p1_accuracy is None or p2_accuracy is None else round(p2_accuracy - p1_accuracy, 4),
                latest_state_correct_rate_delta=None if p1_latest is None or p2_latest is None else round(p2_latest - p1_latest, 4),
                contradiction_resolved_rate_delta=None if p1_resolved is None or p2_resolved is None else round(p2_resolved - p1_resolved, 4),
            )
        )

    return drifts
