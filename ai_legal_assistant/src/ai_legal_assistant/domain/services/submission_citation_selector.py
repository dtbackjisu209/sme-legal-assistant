from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from ai_legal_assistant.domain.entities.competition_submission import (
    LegalCitation,
    ResolvedLegalContext,
)


@dataclass(frozen=True)
class SubmissionCitationSelection:
    relevant_docs: tuple[str, ...]
    relevant_articles: tuple[str, ...]


@dataclass(frozen=True)
class SubmissionCitationSelectorConfig:
    exact_lookup_max_docs: int = 1
    exact_lookup_max_articles: int = 1
    simple_max_docs: int = 2
    simple_max_articles: int = 2
    multi_issue_max_docs: int = 3
    multi_issue_max_articles: int = 5
    min_score: float | None = None
    score_margin: float | None = None


@dataclass(frozen=True)
class _RankedCitation:
    citation: LegalCitation
    score: float | None
    rank: int


class _ScoredHit(Protocol):
    score: float
    metadata: dict[str, Any]


class SubmissionCitationSelector:
    """Keeps submission citations conservative without starving answer context."""

    _EXACT_LOOKUP_PATTERN = re.compile(
        r"\b(?:điều|khoản|điểm)\s+\d|\b\d{1,5}/\d{4}/[\wÀ-ỹĐđ-]+",
        re.IGNORECASE,
    )
    _MULTI_SIGNAL_PATTERNS = tuple(
        re.compile(pattern, re.IGNORECASE)
        for pattern in (
            r"\b(?:như thế nào|ra sao)\s+và\s+",
            r"\bvà\s+(?:phải|cần|bị|được|xử lý|khắc phục|áp dụng|nộp|thực hiện)\b",
            r"\bvề\s+[^?]{3,80}\s+và\s+[^?]{3,80}\?",
            r"\b(?:thuế|đất đai|kế toán|bảo hiểm|xử phạt|khắc phục)\b[^?]*\bvà\b",
        )
    )
    _PROTECTED_AND_PHRASES = (
        "doanh nghiệp nhỏ và vừa",
        "nhỏ và vừa",
        "vừa và nhỏ",
    )

    def __init__(self, config: SubmissionCitationSelectorConfig | None = None) -> None:
        self.config = config or SubmissionCitationSelectorConfig()
        self._validate_config()

    def select(
        self,
        *,
        question: str,
        hits: Sequence[_ScoredHit],
        contexts: Sequence[ResolvedLegalContext],
    ) -> SubmissionCitationSelection:
        max_docs, max_articles = self.limits_for_question(question)
        ranked = self._ranked_citations(hits, contexts)
        ranked = self._apply_score_filters(ranked)

        articles: list[str] = []
        article_citations: list[LegalCitation] = []
        seen_articles: set[str] = set()
        for item in ranked:
            article_entry = item.citation.article_entry
            if article_entry in seen_articles:
                continue
            seen_articles.add(article_entry)
            articles.append(article_entry)
            article_citations.append(item.citation)
            if len(articles) >= max_articles:
                break

        docs = self._select_docs(article_citations, ranked, max_docs=max_docs)
        return SubmissionCitationSelection(
            relevant_docs=tuple(docs),
            relevant_articles=tuple(articles),
        )

    def limits_for_question(self, question: str) -> tuple[int, int]:
        return self._limits(self._profile(question))

    def _validate_config(self) -> None:
        positive_values = {
            "exact_lookup_max_docs": self.config.exact_lookup_max_docs,
            "exact_lookup_max_articles": self.config.exact_lookup_max_articles,
            "simple_max_docs": self.config.simple_max_docs,
            "simple_max_articles": self.config.simple_max_articles,
            "multi_issue_max_docs": self.config.multi_issue_max_docs,
            "multi_issue_max_articles": self.config.multi_issue_max_articles,
        }
        for name, value in positive_values.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive.")
        if self.config.score_margin is not None and self.config.score_margin < 0:
            raise ValueError("score_margin cannot be negative.")

    def _limits(self, profile: str) -> tuple[int, int]:
        if profile == "exact_lookup":
            return (
                self.config.exact_lookup_max_docs,
                self.config.exact_lookup_max_articles,
            )
        if profile == "multi_issue":
            return (
                self.config.multi_issue_max_docs,
                self.config.multi_issue_max_articles,
            )
        return self.config.simple_max_docs, self.config.simple_max_articles

    def _profile(self, question: str) -> str:
        folded = question.casefold()
        if self._EXACT_LOOKUP_PATTERN.search(folded):
            return "exact_lookup"
        cleaned = folded
        for phrase in self._PROTECTED_AND_PHRASES:
            cleaned = cleaned.replace(phrase, phrase.replace(" và ", " "))
        if any(pattern.search(cleaned) for pattern in self._MULTI_SIGNAL_PATTERNS):
            return "multi_issue"
        return "simple"

    def _ranked_citations(
        self,
        hits: Sequence[_ScoredHit],
        contexts: Sequence[ResolvedLegalContext],
    ) -> list[_RankedCitation]:
        ranked: list[_RankedCitation] = []
        for rank, (hit, context) in enumerate(zip(hits, contexts, strict=False), start=1):
            if context.citation is None:
                continue
            ranked.append(
                _RankedCitation(
                    citation=context.citation,
                    score=self._score(hit),
                    rank=rank,
                )
            )
        return sorted(
            ranked,
            key=lambda item: (
                item.score if item.score is not None else -math.inf,
                -item.rank,
            ),
            reverse=True,
        )

    @staticmethod
    def _score(hit: _ScoredHit) -> float | None:
        value = hit.metadata.get("rerank_score", hit.score)
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _apply_score_filters(
        self,
        ranked: list[_RankedCitation],
    ) -> list[_RankedCitation]:
        if not ranked:
            return []
        filtered = ranked
        if self.config.min_score is not None:
            filtered = [
                item
                for item in filtered
                if item.score is not None and item.score >= self.config.min_score
            ]
        if self.config.score_margin is not None and filtered:
            scored = [item.score for item in filtered if item.score is not None]
            if scored:
                floor = max(scored) - self.config.score_margin
                filtered = [
                    item
                    for item in filtered
                    if item.score is not None and item.score >= floor
                ]
        return filtered

    @staticmethod
    def _select_docs(
        article_citations: Sequence[LegalCitation],
        ranked: Sequence[_RankedCitation],
        *,
        max_docs: int,
    ) -> list[str]:
        docs: list[str] = []
        seen: set[str] = set()
        for citation in article_citations:
            entry = citation.document_entry
            if entry not in seen:
                seen.add(entry)
                docs.append(entry)
            if len(docs) >= max_docs:
                return docs
        for item in ranked:
            entry = item.citation.document_entry
            if entry not in seen:
                seen.add(entry)
                docs.append(entry)
            if len(docs) >= max_docs:
                break
        return docs
