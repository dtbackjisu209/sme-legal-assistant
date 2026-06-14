from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class IngestPhapdienCommand:
    output_dir: Path = Path("data/processed")
    limit: int | None = None


@dataclass(frozen=True)
class IngestPhapdienResult:
    article_count: int
    chunk_count: int
    articles_path: Path
    chunks_path: Path

    def __str__(self) -> str:
        return (
            f"Wrote {self.article_count} articles and {self.chunk_count} chunks "
            f"to {self.articles_path.parent}"
        )
