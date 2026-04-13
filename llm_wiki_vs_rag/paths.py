"""Path helpers for deterministic benchmark filesystem layout."""

from pathlib import Path


class ProjectPaths:
    """Encapsulates standard directories and files used by the benchmark."""

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root

    @property
    def raw_dir(self) -> Path:
        return self.project_root / "raw"

    @property
    def wiki_dir(self) -> Path:
        return self.project_root / "wiki"

    @property
    def artifacts_dir(self) -> Path:
        return self.project_root / "artifacts"

    @property
    def index_md(self) -> Path:
        return self.wiki_dir / "index.md"

    @property
    def log_md(self) -> Path:
        return self.artifacts_dir / "log.md"

    def ensure(self) -> None:
        """Create required directories and base files if absent."""
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.wiki_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.index_md.touch(exist_ok=True)
        self.log_md.touch(exist_ok=True)
