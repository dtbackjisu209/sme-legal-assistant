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


class ExpandedRetrievalTest(unittest.TestCase):
    def test_fuses_weighted_query_results_and_deduplicates_chunks(self) -> None:
        retriever = RetrieveExpandedLegalContextUseCase(
            query_planner=FakePlanner(),
            query_embedder=SequentialEmbedder(),
            vector_store=ResultByVectorStore(),
            expected_vector_size=1,
            fusion=WeightedReciprocalRankFusion(rank_constant=60),
            per_query_top_k=2,
        )

        hits = retriever.execute(RetrieveLegalContextQuery("original", top_k=3))

        self.assertEqual([hit.chunk_id for hit in hits], ["c2", "c1", "c3"])
        self.assertEqual(hits[0].text, "two")
        self.assertGreater(hits[0].score, hits[1].score)


if __name__ == "__main__":
    unittest.main()
