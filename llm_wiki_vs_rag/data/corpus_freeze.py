"""Corpus freeze manifest builder and validator."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path


def _parse_order_value(file_name: str) -> tuple[int, int]:
    stem = Path(file_name).stem
    match = re.match(r"^(\d+)", stem)
    if match is None:
        raise ValueError(
            "Corpus freeze requires numbered filenames with a leading numeric prefix. "
            f"invalid={file_name}"
        )
    token = match.group(1)
    return int(token), len(token)


def _infer_phase(relative_path: Path) -> str:
    for part in relative_path.parts:
        normalized = part.lower().replace("-", "_")
        if normalized in {"phase_1", "phase_2"}:
            return normalized
    raise ValueError(f"Corpus freeze could not infer phase from path: {relative_path.as_posix()}")


def _infer_source_type(relative_path: Path) -> str:
    lower_parts = [part.lower() for part in relative_path.parts]
    if "wikipedia" in lower_parts or "wiki" in lower_parts:
        return "wiki"
    if "news" in lower_parts:
        return "news"
    if "policy" in lower_parts:
        return "policy"
    phase = _infer_phase(relative_path)
    phase_index = list(relative_path.parts).index(phase)
    if phase_index + 1 < len(relative_path.parts) - 1:
        return relative_path.parts[phase_index + 1].lower()
    return "unknown"


def _stable_doc_id(relative_path: Path) -> str:
    return Path(relative_path).stem


def build_corpus_manifest(dataset_root: Path) -> list[dict[str, object]]:
    dataset_root = dataset_root.resolve()
    if not dataset_root.exists():
        raise ValueError(f"Dataset root does not exist: {dataset_root}")

    records: list[dict[str, object]] = []
    for file_path in sorted(dataset_root.rglob("*")):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in {".txt", ".md"}:
            continue
        relative_path = file_path.relative_to(dataset_root)
        phase = _infer_phase(relative_path)
        file_name = file_path.name
        order_value, _prefix_width = _parse_order_value(file_name)
        records.append(
            {
                "_phase_order": order_value,
                "file_name": file_name,
                "relative_path": relative_path.as_posix(),
                "source_type": _infer_source_type(relative_path),
                "phase": phase,
                "doc_id": _stable_doc_id(relative_path),
                "is_empty": file_path.read_text(encoding="utf-8") == "",
            }
        )

    if not records:
        raise ValueError("Corpus freeze found no source files under dataset root.")

    phase_rank = {"phase_1": 1, "phase_2": 2}
    records.sort(
        key=lambda item: (
            phase_rank[str(item["phase"])],
            int(item["_phase_order"]),
            str(item["file_name"]),
            str(item["relative_path"]),
        )
    )

    for idx, item in enumerate(records, start=1):
        item["order_index"] = idx
        del item["_phase_order"]

    validate_corpus_manifest(records)
    return records


def validate_corpus_manifest(records: list[dict[str, object]]) -> None:
    phase_1 = [item for item in records if item["phase"] == "phase_1"]
    phase_2 = [item for item in records if item["phase"] == "phase_2"]

    if len(phase_1) != 50:
        raise ValueError(f"Corpus freeze requires exactly 50 files in phase_1, found={len(phase_1)}")
    if len(phase_2) != 50:
        raise ValueError(f"Corpus freeze requires exactly 50 files in phase_2, found={len(phase_2)}")

    doc_ids = [str(item["doc_id"]) for item in records]
    duplicates = sorted(doc_id for doc_id, count in Counter(doc_ids).items() if count > 1)
    if duplicates:
        raise ValueError(f"Corpus freeze requires unique doc_id values, duplicates={duplicates[:5]}")

    order_indexes = [int(item["order_index"]) for item in records]
    if len(set(order_indexes)) != len(order_indexes):
        raise ValueError("Corpus freeze requires order_index to be unique.")
    if order_indexes != list(range(1, len(records) + 1)):
        raise ValueError("Corpus freeze requires order_index to be stable and contiguous from 1..N.")

    for phase in ("phase_1", "phase_2"):
        phase_rows = [item for item in records if item["phase"] == phase]
        numbered = [(_parse_order_value(str(item["file_name"]))[0], str(item["file_name"])) for item in phase_rows]
        if numbered != sorted(numbered, key=lambda value: (value[0], value[1])):
            raise ValueError(
                "Corpus freeze chronology/order validation failed for numbered filenames in "
                f"{phase}."
            )


def write_corpus_manifest(dataset_root: Path, output_dir: Path) -> tuple[Path, Path]:
    records = build_corpus_manifest(dataset_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "manifest.jsonl"
    summary_path = output_dir / "manifest_summary.json"

    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    source_counts = Counter(str(item["source_type"]) for item in records)
    summary = {
        "total_docs": len(records),
        "phase_1_docs": sum(1 for item in records if item["phase"] == "phase_1"),
        "phase_2_docs": sum(1 for item in records if item["phase"] == "phase_2"),
        "source_type_counts": dict(sorted(source_counts.items())),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest_path, summary_path
