from __future__ import annotations

from dataclasses import dataclass
from typing import Any


Posting = tuple[int, int]


@dataclass(frozen=True)
class SearchableLawChunk:
    chunk_id: str
    text: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class BM25Index:
    postings: dict[str, list[Posting]]
    idf: dict[str, float]
    doc_lengths: list[int]
    avgdl: float
    chunk_ids: list[str]
    texts: list[str]
    metadatas: list[dict[str, Any]]
    k1: float = 1.5
    b: float = 0.75
    version: int = 1

    @property
    def document_count(self) -> int:
        return len(self.chunk_ids)

    @property
    def vocabulary_size(self) -> int:
        return len(self.postings)
