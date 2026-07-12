from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from ai_legal_assistant.application.dto.retrieval_dto import DenseSearchResult
from ai_legal_assistant.application.dto.vector_store_dto import VectorPoint


class VectorSearchPort(Protocol):
    def search(self, query_vector: Sequence[float], top_k: int = 10) -> list[DenseSearchResult]:
        ...


class VectorStoreImportPort(Protocol):
    def ensure_collection(self, vector_size: int, recreate: bool = False) -> None:
        ...

    def create_payload_indexes(self) -> None:
        ...

    def upsert(self, points: Sequence[VectorPoint]) -> None:
        ...

    def set_indexing_threshold(self, threshold: int) -> None:
        ...

    def count(self) -> int:
        ...


class VectorStorePort(VectorSearchPort, VectorStoreImportPort, Protocol):
    ...
