from __future__ import annotations

from dataclasses import dataclass

from ai_legal_assistant.application.dto.retrieval_dto import (
    DenseSearchResult,
    RetrievalEvaluationResult,
    RetrieveLegalContextQuery,
)
from ai_legal_assistant.application.ports.retrieval_testset_port import RetrievalTestsetPort
from ai_legal_assistant.application.ports.retriever_port import LegalContextRetrieverPort
from ai_legal_assistant.domain.entities.retrieval_test_case import RetrievalTestCase
from ai_legal_assistant.domain.services.retrieval_metrics import RetrievalMetricsCalculator


@dataclass
class EvaluateRetrievalUseCase:
    retriever: LegalContextRetrieverPort
    testset: RetrievalTestsetPort
    metrics: RetrievalMetricsCalculator

    def execute(self, cutoffs: tuple[int, ...] = (1, 3, 5, 10, 20)) -> RetrievalEvaluationResult:
        normalized_cutoffs = tuple(sorted(set(cutoffs)))
        if not normalized_cutoffs or any(cutoff <= 0 for cutoff in normalized_cutoffs):
            raise ValueError("cutoffs must contain positive integers.")

        totals_recall = {cutoff: 0.0 for cutoff in normalized_cutoffs}
        totals_mrr = {cutoff: 0.0 for cutoff in normalized_cutoffs}
        cases = list(self.testset.load())
        if not cases:
            raise ValueError("Retrieval testset is empty.")

        max_k = max(normalized_cutoffs)
        for case in cases:
            hits = self.retriever.execute(
                RetrieveLegalContextQuery(query=case.query.text, top_k=max_k)
            )
            retrieved_ids = self._relevance_ids(case, hits)
            case_metrics = self.metrics.calculate(
                retrieved_ids=retrieved_ids,
                relevant_ids=case.relevant_ids,
                cutoffs=normalized_cutoffs,
            )
            for cutoff in normalized_cutoffs:
                totals_recall[cutoff] += case_metrics.recall_at_k[cutoff]
                totals_mrr[cutoff] += case_metrics.reciprocal_rank_at_k[cutoff]

        count = len(cases)
        return RetrievalEvaluationResult(
            case_count=count,
            recall_at_k={cutoff: totals_recall[cutoff] / count for cutoff in normalized_cutoffs},
            mrr_at_k={cutoff: totals_mrr[cutoff] / count for cutoff in normalized_cutoffs},
        )

    @staticmethod
    def _relevance_ids(
        case: RetrievalTestCase,
        hits: list[DenseSearchResult],
    ) -> list[str]:
        if case.relevance_field == "chunk_id":
            return [str(getattr(hit, "chunk_id")) for hit in hits]
        return [
            str(getattr(hit, "metadata").get("article_id") or "")
            for hit in hits
        ]
