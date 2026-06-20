from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class CompetitionQuestion:
    """One question supplied by the competition test set."""

    question_id: int
    question: str


@dataclass(frozen=True, slots=True)
class LegalCitation:
    """A legal-document citation in the exact fields required by the dashboard."""

    document_code: str
    document_title: str
    article: str

    @property
    def document_entry(self) -> str:
        return f"{self.document_code}|{self.document_title}"

    @property
    def article_entry(self) -> str:
        return f"{self.document_code}|{self.document_title}|{self.article}"


@dataclass(frozen=True, slots=True)
class ResolvedLegalContext:
    """Retrieved text together with its authoritative citation, when available."""

    text: str
    citation: LegalCitation | None


@dataclass(frozen=True, slots=True)
class SubmissionRecord:
    """A single item in ``results.json``."""

    question_id: int
    question: str
    answer: str
    relevant_docs: tuple[str, ...]
    relevant_articles: tuple[str, ...]

    def to_mapping(self) -> dict[str, Any]:
        return {
            "id": self.question_id,
            "question": self.question,
            "answer": self.answer,
            "relevant_docs": list(self.relevant_docs),
            "relevant_articles": list(self.relevant_articles),
        }
