from __future__ import annotations

from typing import Protocol

from ai_legal_assistant.application.dto.retrieval_dto import SparseSearchResult


class SparseSearchPort(Protocol):
    def search(self, query: str, top_k: int = 10) -> list[SparseSearchResult]:
        ...
