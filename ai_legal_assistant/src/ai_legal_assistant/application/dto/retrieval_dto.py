from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SparseSearchResult:
    chunk_id: str
    score: float
    text: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class DenseSearchResult:
    chunk_id: str
    score: float
    text: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class RetrieveLegalContextQuery:
    query: str
    top_k: int = 10


@dataclass(frozen=True)
class RetrievalEvaluationResult:
    case_count: int
    recall_at_k: dict[int, float]
    mrr_at_k: dict[int, float]


@dataclass(frozen=True)
class BuildBM25IndexCommand:
    chunks_path: Path
    output_path: Path
    limit: int | None = None


@dataclass(frozen=True)
class BuildBM25IndexResult:
    document_count: int
    vocabulary_size: int
    avgdl: float
    output_path: Path

    def __str__(self) -> str:
        return (
            f"Wrote BM25 index with {self.document_count} documents, "
            f"{self.vocabulary_size} terms, avgdl={self.avgdl:.2f} to {self.output_path}"
        )
