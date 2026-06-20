from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ai_legal_assistant.application.dto.retrieval_dto import (
    DenseSearchResult,
    RetrieveLegalContextQuery,
    SparseSearchResult,
)
from ai_legal_assistant.application.use_cases.retrieve_expanded_legal_context import (
    RetrieveExpandedLegalContextUseCase,
)
from ai_legal_assistant.domain.entities.legal_query import LegalQuery
from ai_legal_assistant.domain.entities.query_plan import (
    QueryAnalysis,
    QueryPlan,
    QueryType,
    QueryVariantKind,
    TemporalScope,
    WeightedQuery,
)
from ai_legal_assistant.domain.entities.retrieval_candidate import RerankedCandidate
from ai_legal_assistant.domain.services.weighted_rrf import WeightedReciprocalRankFusion


class FakePlanner:
    def execute(self, raw_query: str) -> QueryPlan:
        original = LegalQuery(raw_query)
        return QueryPlan(
            original_query=original,
            normalized_query=original,
            analysis=QueryAnalysis(
                intent="deadline",
                query_type=QueryType.LEGAL_CONCEPT,
                legal_domain="enterprise",
                entities=("vốn điều lệ",),
                company_type=None,
                article_number=None,
                clause_number=None,
                document_number=None,
                temporal_scope=TemporalScope.CURRENT,
                must_terms=("vốn điều lệ",),
            ),
            semantic_queries=(
                WeightedQuery(raw_query, 1.0, QueryVariantKind.ORIGINAL, "original"),
                WeightedQuery("expanded", 0.7, QueryVariantKind.SEMANTIC, "rewrite"),
            ),
            subqueries=(),
            lexical_terms=("vốn điều lệ",),
        )


class SequentialEmbedder:
    def __init__(self) -> None:
        self.counter = 0

    def embed_query(self, query: LegalQuery) -> list[float]:
        self.counter += 1
        return [float(self.counter)]


class ResultByVectorStore:
    def search(self, query_vector: list[float], top_k: int = 10) -> list[DenseSearchResult]:
        if query_vector == [1.0]:
            return [
                DenseSearchResult("c1", 0.9, "one", {}),
                DenseSearchResult("c2", 0.8, "two", {}),
            ]
        return [
            DenseSearchResult("c2", 0.95, "two", {}),
            DenseSearchResult("c3", 0.7, "three", {}),
        ]


class RecordingSparseSearch:
    def __init__(self) -> None:
        self.queries: list[tuple[str, int]] = []

    def search(self, query: str, top_k: int = 10) -> list[SparseSearchResult]:
        self.queries.append((query, top_k))
        return [SparseSearchResult("c4", 4.0, "sparse only", {})]


class AmbiguousScopePlanner:
    def execute(self, raw_query: str) -> QueryPlan:
        original = LegalQuery(raw_query)
        return QueryPlan(
            original_query=original,
            normalized_query=original,
            analysis=QueryAnalysis(
                intent="deadline",
                query_type=QueryType.AMBIGUOUS,
                legal_domain="enterprise",
                entities=("capital",),
                company_type=None,
                article_number=None,
                clause_number=None,
                document_number=None,
                temporal_scope=TemporalScope.CURRENT,
                must_terms=("capital",),
            ),
            semantic_queries=(
                WeightedQuery(raw_query, 1.0, QueryVariantKind.ORIGINAL, "original"),
                WeightedQuery(
                    "one member capital deadline",
                    0.7,
                    QueryVariantKind.SCOPE,
                    "one",
                    scope="limited_liability_one_member",
                ),
                WeightedQuery(
                    "two member capital deadline",
                    0.7,
                    QueryVariantKind.SCOPE,
                    "two",
                    scope="limited_liability_two_or_more",
                ),
                WeightedQuery(
                    "joint stock capital deadline",
                    0.7,
                    QueryVariantKind.SCOPE,
                    "stock",
                    scope="joint_stock",
                ),
            ),
            subqueries=(),
            lexical_terms=("capital",),
        )


class ScopeVectorStore:
    def search(self, query_vector: list[float], top_k: int = 10) -> list[DenseSearchResult]:
        branch = int(query_vector[0])
        results = {
            1: [DenseSearchResult("noise", 0.99, "member signature list", {"article_id": "n"})],
            2: [
                DenseSearchResult("one", 0.9, "one member must contribute within 90 days", {"article_id": "a1"}),
                DenseSearchResult("one-effect", 0.8, "change capital within 30 days", {"article_id": "a1"}),
            ],
            3: [DenseSearchResult("two", 0.9, "two members must contribute within 90 days", {"article_id": "a2"})],
            4: [DenseSearchResult("stock", 0.9, "shareholders must pay registered shares", {"article_id": "a3"})],
        }
        return results[branch][:top_k]


class FakeIntentReranker:
    def rerank(self, *, query: str, intent: str, scope_queries: dict, candidates: tuple) -> tuple:
        scores = {
            "one member must contribute within 90 days": 0.95,
            "two members must contribute within 90 days": 0.96,
            "shareholders must pay registered shares": 0.94,
            "change capital within 30 days": 0.2,
            "member signature list": 0.1,
        }
        return tuple(
            RerankedCandidate(candidate=item, relevance_score=scores[item.text])
            for item in candidates
        )


class ExpandedRetrievalTest(unittest.TestCase):
    def test_fuses_weighted_query_results_and_deduplicates_chunks(self) -> None:
        retriever = RetrieveExpandedLegalContextUseCase(
            query_planner=FakePlanner(),
            query_embedder=SequentialEmbedder(),
            vector_store=ResultByVectorStore(),
            expected_vector_size=1,
            fusion=WeightedReciprocalRankFusion(rank_constant=60),
            per_query_top_k=3,
        )

        hits = retriever.execute(RetrieveLegalContextQuery("original", top_k=3))

        self.assertEqual([hit.chunk_id for hit in hits], ["c2", "c1", "c3"])
        self.assertEqual(hits[0].text, "two")
        self.assertGreater(hits[0].score, hits[1].score)

    def test_searches_bm25_for_every_semantic_query_and_fuses_sparse_hits(self) -> None:
        sparse_search = RecordingSparseSearch()
        retriever = RetrieveExpandedLegalContextUseCase(
            query_planner=FakePlanner(),
            query_embedder=SequentialEmbedder(),
            vector_store=ResultByVectorStore(),
            expected_vector_size=1,
            fusion=WeightedReciprocalRankFusion(rank_constant=60),
            per_query_top_k=3,
            sparse_search=sparse_search,
            sparse_weight=0.7,
        )

        hits = retriever.execute(RetrieveLegalContextQuery("original", top_k=4))

        self.assertEqual(len(sparse_search.queries), 2)
        self.assertIn("vốn điều lệ", sparse_search.queries[0][0])
        self.assertTrue(all(top_k == 15 for _, top_k in sparse_search.queries))
        self.assertIn("c4", [hit.chunk_id for hit in hits])

    def test_reranks_reserves_each_scope_and_caps_articles(self) -> None:
        retriever = RetrieveExpandedLegalContextUseCase(
            query_planner=AmbiguousScopePlanner(),
            query_embedder=SequentialEmbedder(),
            vector_store=ScopeVectorStore(),
            expected_vector_size=1,
            fusion=WeightedReciprocalRankFusion(rank_constant=60),
            per_query_top_k=5,
            reranker=FakeIntentReranker(),
            oversample_factor=2,
            candidate_pool_size=3,
            scope_candidates_per_branch=2,
            ambiguous_max_chunks_per_article=1,
        )

        hits = retriever.execute(RetrieveLegalContextQuery("capital deadline", top_k=5))

        self.assertEqual([hit.chunk_id for hit in hits], ["two", "one", "stock"])
        self.assertEqual(
            {scope for hit in hits for scope in hit.metadata["matched_scopes"]},
            {
                "limited_liability_one_member",
                "limited_liability_two_or_more",
                "joint_stock",
            },
        )
        self.assertEqual(len({hit.metadata["article_id"] for hit in hits}), 3)
        self.assertTrue(all("content_id" in hit.metadata for hit in hits))


if __name__ == "__main__":
    unittest.main()
