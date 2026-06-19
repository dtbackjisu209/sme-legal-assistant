from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ai_legal_assistant.application.dto.retrieval_dto import DenseSearchResult
from ai_legal_assistant.application.use_cases.evaluate_retrieval import EvaluateRetrievalUseCase
from ai_legal_assistant.domain.entities.legal_query import LegalQuery
from ai_legal_assistant.domain.entities.retrieval_test_case import RetrievalTestCase
from ai_legal_assistant.domain.services.retrieval_metrics import RetrievalMetricsCalculator


class FakeRetriever:
    def execute(self, request: object) -> list[DenseSearchResult]:
        return [
            DenseSearchResult("x", 0.9, "", {"article_id": "wrong"}),
            DenseSearchResult("c1", 0.8, "", {"article_id": "a1"}),
        ]


class FakeTestset:
    def load(self) -> list[RetrievalTestCase]:
        return [
            RetrievalTestCase("1", LegalQuery("q1"), "chunk_id", frozenset({"c1"})),
            RetrievalTestCase("2", LegalQuery("q2"), "article_id", frozenset({"a1"})),
        ]


class EvaluateRetrievalUseCaseTest(unittest.TestCase):
    def test_macro_averages_recall_and_mrr(self) -> None:
        result = EvaluateRetrievalUseCase(
            retriever=FakeRetriever(),
            testset=FakeTestset(),
            metrics=RetrievalMetricsCalculator(),
        ).execute(cutoffs=(1, 2))

        self.assertEqual(result.case_count, 2)
        self.assertEqual(result.recall_at_k, {1: 0.0, 2: 1.0})
        self.assertEqual(result.mrr_at_k, {1: 0.0, 2: 0.5})


if __name__ == "__main__":
    unittest.main()
