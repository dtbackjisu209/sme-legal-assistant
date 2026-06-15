from __future__ import annotations

from typing import Iterable, Protocol

from ai_legal_assistant.domain.entities.bm25_index import BM25Index, SearchableLawChunk


class LawChunkReaderPort(Protocol):
    def iter_chunks(self, limit: int | None = None) -> Iterable[SearchableLawChunk]:
        ...


class BM25IndexStorePort(Protocol):
    def save(self, index: BM25Index) -> None:
        ...

    def load(self) -> BM25Index:
        ...
