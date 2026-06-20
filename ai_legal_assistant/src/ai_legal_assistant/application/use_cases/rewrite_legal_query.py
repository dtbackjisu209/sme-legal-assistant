from __future__ import annotations

from dataclasses import dataclass

from ai_legal_assistant.application.ports.query_planning_port import (
    LegalQueryPlannerPort,
    QueryExpansionRejectedError,
    QueryNormalizerPort,
)
from ai_legal_assistant.domain.entities.legal_query import LegalQuery
from ai_legal_assistant.domain.entities.query_plan import (
    QueryAnalysis,
    QueryPlan,
    QueryType,
    TemporalScope,
)
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
        except ValueError as exc:
            # A local planner can occasionally emit malformed JSON. This must
            # degrade retrieval for one question, not stop a full batch run.
            return self.policy.build_original_only_plan(
                original_query=original_query,
                normalized_query=normalized_query,
                analysis=self._fallback_analysis(),
                warning=f"Query planner output was rejected: {exc}",
            )

        try:
            return self.policy.build_plan(
                original_query=original_query,
                normalized_query=normalized_query,
                analysis=draft.analysis,
                proposal=draft.expansion,
            )
        except ValueError as exc:
            # Preserve a valid analysis when only its proposed expansion is
            # unsafe, then retrieve with the original normalized query.
            return self.policy.build_original_only_plan(
                original_query=original_query,
                normalized_query=normalized_query,
                analysis=draft.analysis,
                warning=f"Query expansion was rejected: {exc}",
            )

    @staticmethod
    def _fallback_analysis() -> QueryAnalysis:
        return QueryAnalysis(
            intent="unknown",
            query_type=QueryType.LEGAL_CONCEPT,
            legal_domain=None,
            entities=(),
            company_type=None,
            article_number=None,
            clause_number=None,
            document_number=None,
            temporal_scope=TemporalScope.CURRENT,
            must_terms=(),
        )
