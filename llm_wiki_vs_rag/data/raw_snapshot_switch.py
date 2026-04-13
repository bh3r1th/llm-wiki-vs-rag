"""Helpers for switching raw corpus snapshot between benchmark phases."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from llm_wiki_vs_rag.paths import ProjectPaths

_ALLOWED_SUFFIXES = {".txt", ".md"}


def _phase_source_files(phase_dir: Path) -> list[Path]:
    return sorted(
        [path for path in phase_dir.iterdir() if path.is_file() and path.suffix.lower() in _ALLOWED_SUFFIXES],
        key=lambda path: path.name,
    )


def switch_phase_corpus(*, paths: ProjectPaths, phase: str, source_root: Path | None = None) -> Path:
    source_base = (source_root or (paths.project_root / "data")).resolve()
    phase_dir = source_base / phase
    if not phase_dir.exists() or not phase_dir.is_dir():
        raise ValueError(f"Missing source phase directory: {phase_dir}")

    source_files = _phase_source_files(phase_dir)
    if not source_files:
        raise ValueError(
            "Source phase snapshot does not contain expected files (.txt/.md): "
            f"phase={phase}, directory={phase_dir}"
        )

    paths.raw_dir.mkdir(parents=True, exist_ok=True)
    for child in sorted(paths.raw_dir.iterdir(), key=lambda path: path.name):
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()

    copied_names: list[str] = []
    for source_file in source_files:
        destination = paths.raw_dir / source_file.name
        shutil.copy2(source_file, destination)
        copied_names.append(source_file.name)

    result_files = sorted(
        [path.name for path in paths.raw_dir.iterdir() if path.is_file() and path.suffix.lower() in _ALLOWED_SUFFIXES]
    )
    if not result_files:
        raise ValueError(f"Raw snapshot switch produced an empty raw corpus: phase={phase}")

    manifest = {
        "phase": phase,
        "copied_count": len(result_files),
        "file_list_summary": {
            "files": result_files,
            "first_file": result_files[0],
            "last_file": result_files[-1],
        },
    }
    manifest_path = paths.artifacts_dir / "raw_snapshot_switch.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest_path
