from __future__ import annotations

import re
from collections.abc import Sequence

from ai_legal_assistant.domain.entities.competition_submission import (
    CompetitionQuestion,
    SubmissionRecord,
)


class SubmissionValidationError(ValueError):
    """Raised when a generated submission cannot be uploaded safely."""


class SubmissionValidator:
    """Validates both the dashboard schema and the supplied test-set coverage."""

    _ARTICLE_PATTERN = re.compile(r"^Điều\s+\S+", re.IGNORECASE)

    def validate(
        self,
        records: Sequence[SubmissionRecord],
        expected_questions: Sequence[CompetitionQuestion],
    ) -> None:
        expected_by_id = self._index_expected_questions(expected_questions)
        if not records:
            raise SubmissionValidationError("Submission must contain at least one result.")

        seen_ids = self._validate_records(records, expected_by_id)
        expected_ids = set(expected_by_id)
        if seen_ids != expected_ids:
            missing = sorted(expected_ids - seen_ids)
            unexpected = sorted(seen_ids - expected_ids)
            details: list[str] = []
            if missing:
                details.append(f"missing ids: {missing[:10]}")
            if unexpected:
                details.append(f"unexpected ids: {unexpected[:10]}")
            raise SubmissionValidationError(
                "Submission ids must exactly cover the test set (" + "; ".join(details) + ")."
            )

    def validate_partial(
        self,
        records: Sequence[SubmissionRecord],
        expected_questions: Sequence[CompetitionQuestion],
    ) -> None:
        """Validate completed checkpoint records without requiring full test-set coverage."""
        expected_by_id = self._index_expected_questions(expected_questions)
        self._validate_records(records, expected_by_id)

    def _validate_records(
        self,
        records: Sequence[SubmissionRecord],
        expected_by_id: dict[int, CompetitionQuestion],
    ) -> set[int]:
        seen_ids: set[int] = set()
        for index, record in enumerate(records):
            location = f"results[{index}]"
            if not isinstance(record.question_id, int) or isinstance(record.question_id, bool):
                raise SubmissionValidationError(f"{location}.id must be an integer.")
            if record.question_id in seen_ids:
                raise SubmissionValidationError(
                    f"Duplicate submission id: {record.question_id}."
                )
            seen_ids.add(record.question_id)

            expected = expected_by_id.get(record.question_id)
            if expected is None:
                raise SubmissionValidationError(
                    f"{location}.id={record.question_id} does not belong to the test set."
                )
            if record.question != expected.question:
                raise SubmissionValidationError(
                    f"{location}.question does not exactly match the supplied test question."
                )
            if not isinstance(record.answer, str) or not record.answer.strip():
                raise SubmissionValidationError(f"{location}.answer must be a non-empty string.")
            self._validate_documents(record.relevant_docs, location)
            self._validate_articles(record.relevant_articles, location)
        return seen_ids

    @staticmethod
    def _index_expected_questions(
        questions: Sequence[CompetitionQuestion],
    ) -> dict[int, CompetitionQuestion]:
        if not questions:
            raise SubmissionValidationError("The supplied test set is empty.")
        indexed: dict[int, CompetitionQuestion] = {}
        for question in questions:
            if question.question_id in indexed:
                raise SubmissionValidationError(
                    f"Duplicate test-set id: {question.question_id}."
                )
            indexed[question.question_id] = question
        return indexed

    def _validate_documents(self, entries: Sequence[str], location: str) -> None:
        self._validate_unique(entries, f"{location}.relevant_docs")
        for entry in entries:
            code, title = self._split(entry, expected_parts=2, field="relevant_docs")
            if code not in title:
                raise SubmissionValidationError(
                    f"{location}.relevant_docs title must include its document code."
                )

    def _validate_articles(self, entries: Sequence[str], location: str) -> None:
        self._validate_unique(entries, f"{location}.relevant_articles")
        for entry in entries:
            code, title, article = self._split(
                entry,
                expected_parts=3,
                field="relevant_articles",
            )
            if code not in title:
                raise SubmissionValidationError(
                    f"{location}.relevant_articles title must include its document code."
                )
            if not self._ARTICLE_PATTERN.match(article):
                raise SubmissionValidationError(
                    f"{location}.relevant_articles must end with a value such as 'Điều 4'."
                )

    @staticmethod
    def _validate_unique(entries: Sequence[str], field: str) -> None:
        if not isinstance(entries, tuple):
            raise SubmissionValidationError(f"{field} must be a list of strings.")
        if len(entries) != len(set(entries)):
            raise SubmissionValidationError(f"{field} must not contain duplicate entries.")

    @staticmethod
    def _split(entry: str, *, expected_parts: int, field: str) -> tuple[str, ...]:
        if not isinstance(entry, str):
            raise SubmissionValidationError(f"{field} entries must be strings.")
        parts = tuple(part.strip() for part in entry.split("|"))
        if len(parts) != expected_parts or any(not part for part in parts):
            raise SubmissionValidationError(
                f"{field} entries must contain exactly {expected_parts} non-empty pipe-separated fields."
            )
        return parts
