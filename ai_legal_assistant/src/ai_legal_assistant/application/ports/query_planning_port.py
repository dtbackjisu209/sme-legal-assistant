from __future__ import annotations

from typing import Protocol

from ai_legal_assistant.domain.entities.legal_query import LegalQuery
from ai_legal_assistant.domain.entities.query_plan import QueryAnalysis, QueryPlanningDraft


class QueryExpansionRejectedError(ValueError):
    def __init__(self, message: str, analysis: QueryAnalysis) -> None:
        super().__init__(message)
        self.analysis = analysis


class QueryNormalizerPort(Protocol):
    def normalize(self, query: LegalQuery) -> LegalQuery:
        ...


class LegalQueryPlannerPort(Protocol):
    def plan(self, query: LegalQuery) -> QueryPlanningDraft:
        ...
