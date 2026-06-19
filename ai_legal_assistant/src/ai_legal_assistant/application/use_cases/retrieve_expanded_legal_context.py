from __future__ import annotations

from dataclasses import dataclass

from ai_legal_assistant.application.dto.retrieval_dto import (
    DenseSearchResult,
    RetrieveLegalContextQuery,
)
from ai_legal_assistant.application.ports.embedding_port import QueryEmbeddingPort
from ai_legal_assistant.application.ports.vector_store_port import VectorSearchPort
from ai_legal_assistant.application.use_cases.rewrite_legal_query import (
    BuildLegalQueryPlanUseCase,
)
from ai_legal_assistant.domain.entities.legal_query import LegalQuery
from ai_legal_assistant.domain.entities.query_plan import QueryPlan, WeightedQuery
from ai_legal_assistant.domain.services.weighted_rrf import (
    WeightedRankedList,
    WeightedReciprocalRankFusion,
)


@dataclass
class RetrieveExpandedLegalContextUseCase:
    query_planner: BuildLegalQueryPlanUseCase
    query_embedder: QueryEmbeddingPort
    vector_store: VectorSearchPort
    expected_vector_size: int
    fusion: WeightedReciprocalRankFusion
    per_query_top_k: int = 20

    def execute(self, request: RetrieveLegalContextQuery) -> list[DenseSearchResult]:
        plan = self.build_plan(request.query)
        return self.execute_plan(plan, top_k=request.top_k)

    def build_plan(self, query: str) -> QueryPlan:
        return self.query_planner.execute(query)

    def execute_plan(self, plan: QueryPlan, *, top_k: int) -> list[DenseSearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        if self.per_query_top_k <= 0:
            raise ValueError("per_query_top_k must be positive.")

        weighted_queries = (*plan.semantic_queries, *plan.subqueries)
        ranked_lists: list[WeightedRankedList] = []
        best_hits: dict[str, DenseSearchResult] = {}
        for weighted_query in weighted_queries:
            hits = self._search(weighted_query)
            ranked_lists.append(
                WeightedRankedList(
                    item_ids=tuple(hit.chunk_id for hit in hits),
                    weight=weighted_query.weight,
                )
            )
            for hit in hits:
                current = best_hits.get(hit.chunk_id)
                if current is None or hit.score > current.score:
                    best_hits[hit.chunk_id] = hit

        fused_scores = self.fusion.fuse(tuple(ranked_lists))
        ranked_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)[:top_k]
        return [
            DenseSearchResult(
                chunk_id=chunk_id,
                score=fused_scores[chunk_id],
                text=best_hits[chunk_id].text,
                metadata=best_hits[chunk_id].metadata,
            )
            for chunk_id in ranked_ids
        ]

    def _search(self, weighted_query: WeightedQuery) -> list[DenseSearchResult]:
        vector = list(self.query_embedder.embed_query(LegalQuery(weighted_query.text)))
        if len(vector) != self.expected_vector_size:
            raise ValueError(
                f"Query embedding has dimension {len(vector)}; "
                f"Qdrant collection expects {self.expected_vector_size}."
            )
        return self.vector_store.search(vector, top_k=self.per_query_top_k)
