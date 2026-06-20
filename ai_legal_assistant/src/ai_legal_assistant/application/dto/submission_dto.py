from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GenerateSubmissionCommand:
    output_dir: Path
    retrieval_top_k: int = 8


@dataclass(frozen=True)
class SubmissionArtifact:
    record_count: int
    results_path: Path
    zip_path: Path
