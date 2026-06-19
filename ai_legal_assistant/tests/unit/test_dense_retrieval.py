from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ai_legal_assistant.application.dto.retrieval_dto import (
    DenseSearchResult,
    RetrieveLegalContextQuery,
)
from ai_legal_assistant.application.use_cases.retrieve_legal_context import RetrieveLegalContextUseCase
from ai_legal_assistant.domain.entities.legal_query import LegalQuery
from ai_legal_assistant.domain.services.retrieval_metrics import RetrievalMetricsCalculator
from ai_legal_assistant.infrastructure.persistence.jsonl_retrieval_testset import (
    JsonlRetrievalTestset,
)
from ai_legal_assistant.infrastructure.embeddings.ollama_query_embedder import (
    OllamaQueryEmbedder,
    OllamaQueryEmbedderConfig,
)
from ai_legal_assistant.infrastructure.embeddings.qwen3_query_embedder import (
    Qwen3QueryEmbedder,
    Qwen3QueryEmbedderConfig,
)


class FakeQueryEmbedder:
    def __init__(self, vector: list[float]) -> None:
        self.vector = vector
        self.queries: list[LegalQuery] = []

    def embed_query(self, query: LegalQuery) -> list[float]:
        self.queries.append(query)
        return self.vector


class FakeVectorStore:
    def __init__(self, hits: list[DenseSearchResult]) -> None:
        self.hits = hits
        self.searches: list[tuple[list[float], int]] = []

    def search(self, query_vector: list[float], top_k: int = 10) -> list[DenseSearchResult]:
        self.searches.append((list(query_vector), top_k))
        return self.hits[:top_k]


class RetrieveLegalContextUseCaseTest(unittest.TestCase):
    def test_embeds_original_query_and_searches_qdrant_port(self) -> None:
        embedder = FakeQueryEmbedder([0.1, 0.2, 0.3])
        expected = [DenseSearchResult("c1", 0.9, "text", {"article_id": "a1"})]
        vector_store = FakeVectorStore(expected)
        use_case = RetrieveLegalContextUseCase(embedder, vector_store, expected_vector_size=3)

        result = use_case.execute(RetrieveLegalContextQuery("  quyền của doanh nghiệp  ", top_k=5))

        self.assertEqual(result, expected)
        self.assertEqual(embedder.queries[0].text, "quyền của doanh nghiệp")
        self.assertEqual(vector_store.searches, [([0.1, 0.2, 0.3], 5)])

    def test_rejects_embedding_dimension_mismatch(self) -> None:
        use_case = RetrieveLegalContextUseCase(
            FakeQueryEmbedder([0.1, 0.2]),
            FakeVectorStore([]),
            expected_vector_size=3,
        )

        with self.assertRaisesRegex(ValueError, "dimension 2"):
            use_case.execute(RetrieveLegalContextQuery("query"))


class RetrievalMetricsCalculatorTest(unittest.TestCase):
    def test_calculates_recall_and_reciprocal_rank_at_each_cutoff(self) -> None:
        result = RetrievalMetricsCalculator().calculate(
            retrieved_ids=["x", "b", "a", "z"],
            relevant_ids=frozenset({"a", "b"}),
            cutoffs=(1, 2, 3),
        )

        self.assertEqual(result.recall_at_k, {1: 0.0, 2: 0.5, 3: 1.0})
        self.assertEqual(result.reciprocal_rank_at_k, {1: 0.0, 2: 0.5, 3: 0.5})


class JsonlRetrievalTestsetTest(unittest.TestCase):
    def test_loads_chunk_and_article_level_cases(self) -> None:
        rows = [
            {"id": "chunk-case", "query": "q1", "relevant_chunk_ids": ["c1", "c2"]},
            {"id": "article-case", "query": "q2", "relevant_article_ids": ["a1"]},
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "testset.jsonl"
            path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )
            cases = list(JsonlRetrievalTestset(path).load())

        self.assertEqual(cases[0].relevance_field, "chunk_id")
        self.assertEqual(cases[0].relevant_ids, frozenset({"c1", "c2"}))
        self.assertEqual(cases[1].relevance_field, "article_id")

    def test_rejects_ambiguous_relevance_levels(self) -> None:
        row = {
            "query": "q",
            "relevant_chunk_ids": ["c1"],
            "relevant_article_ids": ["a1"],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "testset.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Exactly one"):
                list(JsonlRetrievalTestset(path).load())


class Qwen3QueryEmbedderFormattingTest(unittest.TestCase):
    def test_adds_retrieval_instruction_without_expanding_query(self) -> None:
        embedder = object.__new__(Qwen3QueryEmbedder)
        embedder.config = Qwen3QueryEmbedderConfig(
            model_name_or_path="model",
            query_instruction="Retrieve Vietnamese legal passages",
        )

        formatted = embedder._format_query("Thời hạn góp vốn là bao lâu?")

        self.assertEqual(
            formatted,
            "Instruct: Retrieve Vietnamese legal passages\n"
            "Query:Thời hạn góp vốn là bao lâu?",
        )


class FakeHttpResponse:
    def __enter__(self) -> "FakeHttpResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return b'{"embeddings":[[0.1,0.2,0.3]]}'


class OllamaQueryEmbedderTest(unittest.TestCase):
    @patch(
        "ai_legal_assistant.infrastructure.embeddings.ollama_query_embedder.urlopen",
        return_value=FakeHttpResponse(),
    )
    def test_calls_ollama_embed_api(self, mocked_urlopen: object) -> None:
        embedder = OllamaQueryEmbedder(
            OllamaQueryEmbedderConfig(
                model="qwen3-embedding:0.6b",
                query_instruction=None,
            )
        )

        vector = embedder.embed_query(LegalQuery("Thời hạn góp vốn?"))

        self.assertEqual(vector, [0.1, 0.2, 0.3])
        request = mocked_urlopen.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(request.full_url, "http://localhost:11434/api/embed")
        self.assertEqual(payload["model"], "qwen3-embedding:0.6b")
        self.assertEqual(payload["input"], "Thời hạn góp vốn?")


if __name__ == "__main__":
    unittest.main()
