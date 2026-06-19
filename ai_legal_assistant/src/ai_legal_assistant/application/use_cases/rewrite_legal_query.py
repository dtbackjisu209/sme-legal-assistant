from __future__ import annotations

from dataclasses import dataclass

from ai_legal_assistant.application.ports.query_planning_port import (
    LegalQueryPlannerPort,
    QueryExpansionRejectedError,
    QueryNormalizerPort,
)
from ai_legal_assistant.domain.entities.legal_query import LegalQuery
from ai_legal_assistant.domain.entities.query_plan import QueryPlan
from ai_legal_assistant.domain.services.query_expansion_policy import QueryExpansionPolicy


@dataclass
class BuildLegalQueryPlanUseCase:
    normalizer: QueryNormalizerPort
    planner: LegalQueryPlannerPort
    policy: QueryExpansionPolicy

    def execute(self, raw_query: str) -> QueryPlan:
        original_query = LegalQuery(raw_query)
        normalized_query = self.normalizer.normalize(original_query)
        try:
            draft = self.planner.plan(normalized_query)
        except QueryExpansionRejectedError as exc:
            return self.policy.build_original_only_plan(
                original_query=original_query,
                normalized_query=normalized_query,
                analysis=exc.analysis,
                warning=str(exc),
            )
        return self.policy.build_plan(
            original_query=original_query,
            normalized_query=normalized_query,
            analysis=draft.analysis,
            proposal=draft.expansion,
        )
