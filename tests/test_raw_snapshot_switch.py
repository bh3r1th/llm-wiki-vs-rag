from __future__ import annotations

import json

from llm_wiki_vs_rag.config import AppConfig
from llm_wiki_vs_rag.runner import run_command


def _write_phase_file(root, phase: str, name: str, content: str) -> None:
    phase_dir = root / phase
    phase_dir.mkdir(parents=True, exist_ok=True)
    (phase_dir / name).write_text(content, encoding="utf-8")


def test_switching_to_phase_1_clears_old_raw_files(tmp_path):
    source_root = tmp_path / "snapshots"
    _write_phase_file(source_root, "phase_1", "001_alpha.txt", "phase-1-a")
    _write_phase_file(source_root, "phase_1", "002_beta.txt", "phase-1-b")
    _write_phase_file(source_root, "phase_2", "001_gamma.txt", "phase-2-a")

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "999_old.txt").write_text("stale", encoding="utf-8")

    run_command(
        "switch-phase-corpus",
        AppConfig(project_root=tmp_path),
        phase="phase_1",
        source_root=str(source_root),
    )

    assert sorted(path.name for path in raw_dir.iterdir()) == ["001_alpha.txt", "002_beta.txt"]
    manifest = json.loads((tmp_path / "artifacts" / "raw_snapshot_switch.json").read_text(encoding="utf-8"))
    assert manifest["phase"] == "phase_1"
    assert manifest["copied_count"] == 2


def test_switching_to_phase_2_clears_old_raw_files(tmp_path):
    source_root = tmp_path / "snapshots"
    _write_phase_file(source_root, "phase_1", "001_alpha.txt", "phase-1-a")
    _write_phase_file(source_root, "phase_2", "001_gamma.txt", "phase-2-a")
    _write_phase_file(source_root, "phase_2", "002_delta.md", "phase-2-b")

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "001_alpha.txt").write_text("old-phase-1", encoding="utf-8")

    run_command(
        "switch-phase-corpus",
        AppConfig(project_root=tmp_path),
        phase="phase_2",
        source_root=str(source_root),
    )

    assert sorted(path.name for path in raw_dir.iterdir()) == ["001_gamma.txt", "002_delta.md"]
    assert not (raw_dir / "001_alpha.txt").exists()


def test_mixed_old_new_raw_state_cannot_remain_after_switch(tmp_path):
    source_root = tmp_path / "snapshots"
    _write_phase_file(source_root, "phase_1", "001_new.txt", "new")

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "001_old.txt").write_text("old", encoding="utf-8")
    (raw_dir / "nested").mkdir(parents=True, exist_ok=True)
    (raw_dir / "nested" / "stale.txt").write_text("stale", encoding="utf-8")

    run_command(
        "switch-phase-corpus",
        AppConfig(project_root=tmp_path),
        phase="phase_1",
        source_root=str(source_root),
    )

    assert sorted(path.name for path in raw_dir.iterdir()) == ["001_new.txt"]


def test_missing_source_phase_fails_clearly(tmp_path):
    source_root = tmp_path / "snapshots"
    _write_phase_file(source_root, "phase_1", "001_alpha.txt", "phase-1-a")

    try:
        run_command(
            "switch-phase-corpus",
            AppConfig(project_root=tmp_path),
            phase="phase_2",
            source_root=str(source_root),
        )
    except ValueError as exc:
        assert "Missing source phase directory" in str(exc)
        assert "phase_2" in str(exc)
    else:
        raise AssertionError("Expected missing phase source to fail.")
