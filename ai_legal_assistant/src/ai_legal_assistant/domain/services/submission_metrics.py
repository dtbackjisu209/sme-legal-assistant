from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ai_legal_assistant.domain.entities.competition_submission import SubmissionRecord


@dataclass(frozen=True)
class SubmissionFieldMetrics:
    precision: float
    recall: float
    f2: float


@dataclass(frozen=True)
class SubmissionEvaluationResult:
    case_count: int
    docs: SubmissionFieldMetrics
    articles: SubmissionFieldMetrics


class SubmissionMetricsCalculator:
    def evaluate(
        self,
        *,
        predictions: Sequence[SubmissionRecord],
        gold: Sequence[SubmissionRecord],
    ) -> SubmissionEvaluationResult:
        if not gold:
            raise ValueError("gold records cannot be empty.")
        predictions_by_id = {record.question_id: record for record in predictions}
        gold_by_id = {record.question_id: record for record in gold}
        if len(gold_by_id) != len(gold):
            raise ValueError("gold records contain duplicate ids.")
        if len(predictions_by_id) != len(predictions):
            raise ValueError("prediction records contain duplicate ids.")

        doc_scores: list[SubmissionFieldMetrics] = []
        article_scores: list[SubmissionFieldMetrics] = []
        for question_id, expected in gold_by_id.items():
            predicted = predictions_by_id.get(question_id)
            predicted_docs = set(predicted.relevant_docs) if predicted is not None else set()
            predicted_articles = (
                set(predicted.relevant_articles) if predicted is not None else set()
            )
            doc_scores.append(self._field_metrics(predicted_docs, set(expected.relevant_docs)))
            article_scores.append(
                self._field_metrics(predicted_articles, set(expected.relevant_articles))
            )

        return SubmissionEvaluationResult(
            case_count=len(gold_by_id),
            docs=self._macro_average(doc_scores),
            articles=self._macro_average(article_scores),
        )

    @classmethod
    def _field_metrics(
        cls,
        predicted: set[str],
        relevant: set[str],
    ) -> SubmissionFieldMetrics:
        if not predicted and not relevant:
            return SubmissionFieldMetrics(precision=1.0, recall=1.0, f2=1.0)
        true_positive_count = len(predicted & relevant)
        precision = true_positive_count / len(predicted) if predicted else 0.0
        recall = true_positive_count / len(relevant) if relevant else 0.0
        return SubmissionFieldMetrics(
            precision=precision,
            recall=recall,
            f2=cls._f2(precision, recall),
        )

    @staticmethod
    def _f2(precision: float, recall: float) -> float:
        denominator = (4 * precision) + recall
        if denominator == 0:
            return 0.0
        return (5 * precision * recall) / denominator

    @staticmethod
    def _macro_average(scores: Sequence[SubmissionFieldMetrics]) -> SubmissionFieldMetrics:
        count = len(scores)
        return SubmissionFieldMetrics(
            precision=sum(score.precision for score in scores) / count,
            recall=sum(score.recall for score in scores) / count,
            f2=sum(score.f2 for score in scores) / count,
        )
