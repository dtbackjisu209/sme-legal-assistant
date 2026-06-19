from __future__ import annotations

from typing import Protocol

from ai_legal_assistant.application.dto.retrieval_dto import (
    DenseSearchResult,
    RetrieveLegalContextQuery,
)


class LegalContextRetrieverPort(Protocol):
    def execute(self, request: RetrieveLegalContextQuery) -> list[DenseSearchResult]:
        ...
