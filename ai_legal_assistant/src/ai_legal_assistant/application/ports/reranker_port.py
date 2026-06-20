from __future__ import annotations

from typing import Protocol

from ai_legal_assistant.domain.entities.retrieval_candidate import (
    RerankedCandidate,
    RetrievalCandidate,
)


class CandidateRerankerPort(Protocol):
    def rerank(
        self,
        *,
        query: str,
        intent: str,
        scope_queries: dict[str, str],
        candidates: tuple[RetrievalCandidate, ...],
    ) -> tuple[RerankedCandidate, ...]:
        ...
