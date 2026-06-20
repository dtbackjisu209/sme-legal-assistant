from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from ai_legal_assistant.application.dto.retrieval_dto import (
    DenseSearchResult,
    RetrieveLegalContextQuery,
    SparseSearchResult,
)
from ai_legal_assistant.application.ports.embedding_port import QueryEmbeddingPort
from ai_legal_assistant.application.ports.reranker_port import CandidateRerankerPort
from ai_legal_assistant.application.ports.sparse_search_port import SparseSearchPort
from ai_legal_assistant.application.ports.vector_store_port import VectorSearchPort
from ai_legal_assistant.application.use_cases.rewrite_legal_query import (
    BuildLegalQueryPlanUseCase,
)
from ai_legal_assistant.domain.entities.legal_query import LegalQuery
from ai_legal_assistant.domain.entities.query_plan import QueryPlan, QueryType, WeightedQuery
from ai_legal_assistant.domain.entities.retrieval_candidate import (
    RerankedCandidate,
    RetrievalCandidate,
)
from ai_legal_assistant.domain.services.retrieval_candidate_selector import (
    RetrievalCandidateSelector,
)
from ai_legal_assistant.domain.services.retrieval_content_identity import (
    RetrievalContentIdentity,
)
from ai_legal_assistant.domain.services.weighted_rrf import (
    WeightedRankedList,
    WeightedReciprocalRankFusion,
)


SearchHit = DenseSearchResult | SparseSearchResult


@dataclass
class RetrieveExpandedLegalContextUseCase:
    query_planner: BuildLegalQueryPlanUseCase
    query_embedder: QueryEmbeddingPort
    vector_store: VectorSearchPort
    expected_vector_size: int
    fusion: WeightedReciprocalRankFusion
    per_query_top_k: int = 20
    sparse_search: SparseSearchPort | None = None
    sparse_weight: float = 0.7
    reranker: CandidateRerankerPort | None = None
    oversample_factor: int = 5
    candidate_pool_size: int = 50
    scope_candidates_per_branch: int = 5
    max_chunks_per_article: int = 2
    ambiguous_max_chunks_per_article: int = 1
    scope_relevance_margin: float = 0.15
    content_identity: RetrievalContentIdentity = field(
        default_factory=RetrievalContentIdentity
    )
    candidate_selector: RetrievalCandidateSelector = field(
        default_factory=RetrievalCandidateSelector
    )

    def execute(self, request: RetrieveLegalContextQuery) -> list[DenseSearchResult]:
        plan = self.build_plan(request.query)
        return self.execute_plan(plan, top_k=request.top_k)

    def build_plan(self, query: str) -> QueryPlan:
        return self.query_planner.execute(query)

    def execute_plan(self, plan: QueryPlan, *, top_k: int) -> list[DenseSearchResult]:
        self._validate(top_k)
        raw_top_k = self.per_query_top_k * self.oversample_factor
        weighted_queries = (*plan.semantic_queries, *plan.subqueries)
        representatives: dict[str, DenseSearchResult] = {}
        legacy_ids: dict[str, set[str]] = defaultdict(set)
        matched_scopes: dict[str, set[str]] = defaultdict(set)
        scope_rankings: dict[str, tuple[str, ...]] = {}
        query_ranked_lists: list[WeightedRankedList] = []

        for weighted_query in weighted_queries:
            dense_ids = self._unique_content_ids(
                self._dense_search(weighted_query, top_k=raw_top_k),
                representatives=representatives,
                legacy_ids=legacy_ids,
                limit=self.per_query_top_k,
            )
            modality_lists = [WeightedRankedList(item_ids=dense_ids, weight=1.0)]

            if self.sparse_search is not None:
                sparse_ids = self._unique_content_ids(
                    self.sparse_search.search(
                        self._sparse_query(weighted_query.text, plan.lexical_terms),
                        top_k=raw_top_k,
                    ),
                    representatives=representatives,
                    legacy_ids=legacy_ids,
                    limit=self.per_query_top_k,
                )
                modality_lists.append(
                    WeightedRankedList(item_ids=sparse_ids, weight=self.sparse_weight)
                )

            local_scores = self.fusion.fuse(tuple(modality_lists))
            local_ranked_ids = tuple(
                sorted(local_scores, key=local_scores.get, reverse=True)[
                    : self.per_query_top_k
                ]
            )
            query_ranked_lists.append(
                WeightedRankedList(
                    item_ids=local_ranked_ids,
                    weight=weighted_query.weight,
                )
            )
            if weighted_query.scope:
                scope_rankings[weighted_query.scope] = local_ranked_ids
                for content_id in local_ranked_ids:
                    matched_scopes[content_id].add(weighted_query.scope)

        global_scores = self.fusion.fuse(tuple(query_ranked_lists))
        pool_ids = list(
            sorted(global_scores, key=global_scores.get, reverse=True)[
                : self.candidate_pool_size
            ]
        )
        for scope_ids in scope_rankings.values():
            for content_id in scope_ids[: self.scope_candidates_per_branch]:
                if content_id not in pool_ids:
                    pool_ids.append(content_id)

        candidates = tuple(
            self._candidate(
                content_id,
                representative=representatives[content_id],
                legacy_ids=legacy_ids[content_id],
                matched_scopes=matched_scopes[content_id],
                rrf_score=global_scores.get(content_id, 0.0),
            )
            for content_id in pool_ids
        )
        reranked = self._rerank(plan, candidates)
        scope_queries = {
            query.scope: query.text
            for query in plan.semantic_queries
            if query.scope is not None
        }
        required_scopes = (
            tuple(scope_queries)
            if plan.analysis.query_type == QueryType.AMBIGUOUS
            else ()
        )
        article_cap = (
            self.ambiguous_max_chunks_per_article
            if required_scopes
            else self.max_chunks_per_article
        )
        selected = self.candidate_selector.select(
            reranked,
            top_k=top_k,
            required_scopes=required_scopes,
            max_chunks_per_article=article_cap,
            restrict_to_required_scopes=bool(required_scopes),
            scope_relevance_margin=(
                self.scope_relevance_margin
                if required_scopes and self.reranker is not None
                else None
            ),
        )
        return [self._to_result(item) for item in selected]

    def _validate(self, top_k: int) -> None:
        positive_values = {
            "top_k": top_k,
            "per_query_top_k": self.per_query_top_k,
            "oversample_factor": self.oversample_factor,
            "candidate_pool_size": self.candidate_pool_size,
            "scope_candidates_per_branch": self.scope_candidates_per_branch,
            "max_chunks_per_article": self.max_chunks_per_article,
            "ambiguous_max_chunks_per_article": self.ambiguous_max_chunks_per_article,
            "scope_relevance_margin": self.scope_relevance_margin,
        }
        for name, value in positive_values.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive.")
        if not 0 < self.sparse_weight <= 1:
            raise ValueError("sparse_weight must be in the interval (0, 1].")

    def _dense_search(
        self,
        weighted_query: WeightedQuery,
        *,
        top_k: int,
    ) -> list[DenseSearchResult]:
        vector = list(self.query_embedder.embed_query(LegalQuery(weighted_query.text)))
        if len(vector) != self.expected_vector_size:
            raise ValueError(
                f"Query embedding has dimension {len(vector)}; "
                f"Qdrant collection expects {self.expected_vector_size}."
            )
        return self.vector_store.search(vector, top_k=top_k)

    def _unique_content_ids(
        self,
        hits: list[DenseSearchResult] | list[SparseSearchResult],
        *,
        representatives: dict[str, DenseSearchResult],
        legacy_ids: dict[str, set[str]],
        limit: int,
    ) -> tuple[str, ...]:
        unique_ids: list[str] = []
        seen: set[str] = set()
        for hit in hits:
            article_id = str(hit.metadata.get("article_id") or "")
            content_id = self.content_identity.create(
                article_id=article_id,
                text=hit.text,
            )
            legacy_ids[content_id].add(hit.chunk_id)
            representatives.setdefault(
                content_id,
                DenseSearchResult(
                    chunk_id=hit.chunk_id,
                    score=hit.score,
                    text=hit.text,
                    metadata=dict(hit.metadata),
                ),
            )
            if content_id in seen:
                continue
            seen.add(content_id)
            unique_ids.append(content_id)
            if len(unique_ids) >= limit:
                break
        return tuple(unique_ids)

    def _candidate(
        self,
        content_id: str,
        *,
        representative: DenseSearchResult,
        legacy_ids: set[str],
        matched_scopes: set[str],
        rrf_score: float,
    ) -> RetrievalCandidate:
        metadata = dict(representative.metadata)
        metadata.update(
            {
                "content_id": content_id,
                "legacy_chunk_ids": sorted(legacy_ids),
                "matched_scopes": sorted(matched_scopes),
                "rrf_score": rrf_score,
            }
        )
        return RetrievalCandidate(
            content_id=content_id,
            legacy_chunk_ids=tuple(sorted(legacy_ids)),
            text=representative.text,
            metadata=metadata,
            rrf_score=rrf_score,
            matched_scopes=tuple(sorted(matched_scopes)),
        )

    def _rerank(
        self,
        plan: QueryPlan,
        candidates: tuple[RetrievalCandidate, ...],
    ) -> tuple[RerankedCandidate, ...]:
        if self.reranker is None:
            return tuple(
                RerankedCandidate(candidate=item, relevance_score=item.rrf_score)
                for item in candidates
            )
        return self.reranker.rerank(
            query=plan.normalized_query.text,
            intent=plan.analysis.intent,
            scope_queries={
                query.scope: query.text
                for query in plan.semantic_queries
                if query.scope is not None
            },
            candidates=candidates,
        )

    @staticmethod
    def _sparse_query(query: str, lexical_terms: tuple[str, ...]) -> str:
        query_folded = query.casefold()
        additions = [term for term in lexical_terms if term.casefold() not in query_folded]
        return " ".join((query, *additions))

    @staticmethod
    def _to_result(item: RerankedCandidate) -> DenseSearchResult:
        metadata = dict(item.candidate.metadata)
        metadata["rerank_score"] = item.relevance_score
        legacy_id = (
            item.candidate.legacy_chunk_ids[0]
            if item.candidate.legacy_chunk_ids
            else item.candidate.content_id
        )
        return DenseSearchResult(
            chunk_id=legacy_id,
            score=item.relevance_score,
            text=item.candidate.text,
            metadata=metadata,
        )
