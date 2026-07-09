from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from ai_legal_assistant.application.dto.retrieval_dto import DenseSearchResult
from ai_legal_assistant.domain.entities.competition_submission import (
    LegalCitation,
    ResolvedLegalContext,
    SubmissionRecord,
)
from ai_legal_assistant.domain.services.submission_citation_selector import (
    SubmissionCitationSelector,
    SubmissionCitationSelectorConfig,
)
from ai_legal_assistant.domain.services.submission_metrics import SubmissionMetricsCalculator
from postprocess_submission_citations import filter_records


def _citation(index: int, *, doc: str | None = None) -> LegalCitation:
    document_code = doc or f"{index:02d}/2020/QH14"
    return LegalCitation(
        document_code=document_code,
        document_title=f"Luật {document_code} thử nghiệm",
        article=f"Điều {index}",
    )


def _hit(index: int, score: float) -> DenseSearchResult:
    return DenseSearchResult(
        chunk_id=f"chunk-{index}",
        score=score,
        text=f"text {index}",
        metadata={"rerank_score": score},
    )


class SubmissionCitationSelectorTest(unittest.TestCase):
    def test_simple_question_keeps_only_top_two_articles(self) -> None:
        selector = SubmissionCitationSelector()
        hits = [_hit(index, 1.0 - (index / 10)) for index in range(1, 5)]
        contexts = [
            ResolvedLegalContext(text=hit.text, citation=_citation(index))
            for index, hit in enumerate(hits, start=1)
        ]

        result = selector.select(
            question="Doanh nghiệp được ưu đãi gì khi tham gia đấu thầu?",
            hits=hits,
            contexts=contexts,
        )

        self.assertEqual(len(result.relevant_articles), 2)
        self.assertIn("Điều 1", result.relevant_articles[0])
        self.assertIn("Điều 2", result.relevant_articles[1])
        self.assertEqual(len(result.relevant_docs), 2)

    def test_multi_issue_question_allows_more_articles(self) -> None:
        selector = SubmissionCitationSelector()
        hits = [_hit(index, 1.0 - (index / 20)) for index in range(1, 7)]
        contexts = [
            ResolvedLegalContext(text=hit.text, citation=_citation(index, doc="01/2020/QH14"))
            for index, hit in enumerate(hits, start=1)
        ]

        result = selector.select(
            question="Công ty bị xử lý như thế nào và phải khắc phục ra sao?",
            hits=hits,
            contexts=contexts,
        )

        self.assertEqual(len(result.relevant_articles), 5)
        self.assertEqual(len(result.relevant_docs), 1)

    def test_score_margin_filters_weak_citations(self) -> None:
        selector = SubmissionCitationSelector(
            SubmissionCitationSelectorConfig(simple_max_articles=5, score_margin=0.05)
        )
        hits = [_hit(1, 0.90), _hit(2, 0.84), _hit(3, 0.70)]
        contexts = [
            ResolvedLegalContext(text=hit.text, citation=_citation(index))
            for index, hit in enumerate(hits, start=1)
        ]

        result = selector.select(question="Câu hỏi pháp lý chung?", hits=hits, contexts=contexts)

        self.assertEqual(len(result.relevant_articles), 1)
        self.assertIn("Điều 1", result.relevant_articles[0])


class SubmissionMetricsCalculatorTest(unittest.TestCase):
    def test_macro_averages_docs_and_articles_f2(self) -> None:
        predictions = [
            SubmissionRecord(1, "", "", ("d1", "d2"), ("a1", "a2")),
            SubmissionRecord(2, "", "", (), ("a3",)),
        ]
        gold = [
            SubmissionRecord(1, "", "", ("d1",), ("a1",)),
            SubmissionRecord(2, "", "", ("d3",), ("a3", "a4")),
        ]

        result = SubmissionMetricsCalculator().evaluate(
            predictions=predictions,
            gold=gold,
        )

        self.assertEqual(result.case_count, 2)
        self.assertAlmostEqual(result.docs.precision, 0.25)
        self.assertAlmostEqual(result.docs.recall, 0.5)
        self.assertAlmostEqual(result.articles.precision, 0.75)
        self.assertAlmostEqual(result.articles.recall, 0.75)
        self.assertAlmostEqual(result.articles.f2, 0.6944444444444444)


class PostprocessSubmissionCitationsTest(unittest.TestCase):
    def test_filters_existing_results_without_touching_answer(self) -> None:
        record = SubmissionRecord(
            question_id=1,
            question="Doanh nghiệp được ưu đãi gì khi tham gia đấu thầu?",
            answer="Giữ nguyên câu trả lời.",
            relevant_docs=(
                "01/2020/QH14|Luật 01/2020/QH14 thử nghiệm",
                "02/2020/QH14|Luật 02/2020/QH14 thử nghiệm",
                "03/2020/QH14|Luật 03/2020/QH14 thử nghiệm",
            ),
            relevant_articles=(
                "01/2020/QH14|Luật 01/2020/QH14 thử nghiệm|Điều 1",
                "02/2020/QH14|Luật 02/2020/QH14 thử nghiệm|Điều 2",
                "03/2020/QH14|Luật 03/2020/QH14 thử nghiệm|Điều 3",
                "04/2020/QH14|Luật 04/2020/QH14 thử nghiệm|Điều 4",
            ),
        )

        filtered = filter_records([record], selector=SubmissionCitationSelector())

        self.assertEqual(filtered[0].answer, record.answer)
        self.assertEqual(len(filtered[0].relevant_docs), 2)
        self.assertEqual(len(filtered[0].relevant_articles), 2)
        self.assertIn("Điều 1", filtered[0].relevant_articles[0])
        self.assertIn("Điều 2", filtered[0].relevant_articles[1])


if __name__ == "__main__":
    unittest.main()
