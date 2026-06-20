from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ai_legal_assistant.application.dto.retrieval_dto import DenseSearchResult
from ai_legal_assistant.domain.entities.competition_submission import (
    LegalCitation,
    ResolvedLegalContext,
)


@dataclass(frozen=True, slots=True)
class _CitationSeed:
    document_code: str
    document_type: str
    summary: str
    article: str


class JsonlLegalCitationResolver:
    """Recovers official document and article citations from the Pháp điển corpus.

    Chunks in Qdrant intentionally contain only retrieval metadata.  The authoritative
    document number and trích yếu live in ``law_articles.jsonl``'s source note, so this
    adapter builds a small citation-only index once at startup rather than asking the
    answer model to invent references.
    """

    _DOCUMENT_TYPES = (
        "Nghị quyết liên tịch",
        "Thông tư liên tịch",
        "Văn bản hợp nhất",
        "Bộ luật",
        "Pháp lệnh",
        "Nghị định",
        "Thông tư",
        "Quyết định",
        "Chỉ thị",
        "Nghị quyết",
        "Luật",
    )
    _TYPE_PATTERN = "|".join(re.escape(item) for item in _DOCUMENT_TYPES)
    _DOCUMENT_PATTERN = re.compile(
        rf"(?P<type>{_TYPE_PATTERN})\s+(?:số\s+)?"
        r"(?P<code>\d{1,5}(?:/\d{4})?/[\wÀ-ỹĐđ]+(?:-[\wÀ-ỹĐđ]+)*)"
        r"(?P<summary>.*?)"
        r"(?=\s+ngày\s+\d{1,2}[/-]\d{1,2}[/-]\d{4}\b|\s+của\s+|,|\)|$)",
        re.IGNORECASE | re.DOTALL,
    )
    _ARTICLE_PATTERN = re.compile(
        r"\bĐiều\s+(?P<number>\d+[A-Za-zÀ-ỹĐđ]*(?:\.\d+)*)\b",
        re.IGNORECASE,
    )
    _SUMMARY_CLEANUP = re.compile(r"\s+")

    def __init__(self, articles_path: Path) -> None:
        self.articles_path = articles_path
        self._citation_by_article_id: dict[str, LegalCitation] | None = None

    def resolve(self, hits: Sequence[DenseSearchResult]) -> list[ResolvedLegalContext]:
        citation_by_article_id = self._load_citation_index()
        contexts: list[ResolvedLegalContext] = []
        for hit in hits:
            article_id = str(hit.metadata.get("article_id") or "")
            citation = citation_by_article_id.get(article_id)
            contexts.append(ResolvedLegalContext(text=hit.text, citation=citation))
        return contexts

    def _load_citation_index(self) -> dict[str, LegalCitation]:
        if self._citation_by_article_id is not None:
            return self._citation_by_article_id
        if not self.articles_path.is_file():
            raise ValueError(f"Law article corpus does not exist: {self.articles_path}")

        raw_seeds: dict[str, _CitationSeed] = {}
        summaries_by_document: dict[tuple[str, str], list[str]] = defaultdict(list)
        with self.articles_path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON on line {line_number} of {self.articles_path}."
                    ) from exc
                article_id = row.get("article_id")
                if not isinstance(article_id, str) or not article_id:
                    continue
                seed = self._seed_from_row(row)
                if seed is None:
                    continue
                raw_seeds[article_id] = seed
                if seed.summary:
                    summaries_by_document[(seed.document_type, seed.document_code)].append(
                        seed.summary
                    )

        best_summary = {
            key: self._best_summary(summaries)
            for key, summaries in summaries_by_document.items()
        }
        self._citation_by_article_id = {
            article_id: LegalCitation(
                document_code=seed.document_code,
                document_title=self._document_title(
                    seed,
                    best_summary.get((seed.document_type, seed.document_code), ""),
                ),
                article=seed.article,
            )
            for article_id, seed in raw_seeds.items()
        }
        return self._citation_by_article_id

    def _seed_from_row(self, row: dict[str, Any]) -> _CitationSeed | None:
        note = row.get("source_note_text")
        if not isinstance(note, str):
            return None
        document_match = self._DOCUMENT_PATTERN.search(note)
        article_match = self._ARTICLE_PATTERN.search(note)
        if document_match is None or article_match is None:
            return None
        document_type = self._normalise_document_type(document_match.group("type"))
        document_code = self._normalise_whitespace(document_match.group("code"))
        summary = self._normalise_summary(document_match.group("summary"))
        return _CitationSeed(
            document_code=document_code,
            document_type=document_type,
            summary=summary,
            article=f"Điều {article_match.group('number')}",
        )

    @classmethod
    def _normalise_document_type(cls, value: str) -> str:
        folded = value.casefold()
        for item in cls._DOCUMENT_TYPES:
            if item.casefold() == folded:
                return item
        raise ValueError(f"Unsupported document type: {value}")

    @classmethod
    def _normalise_summary(cls, value: str) -> str:
        summary = cls._normalise_whitespace(value).strip(" ,;:-()")
        for marker in (" có hiệu lực", " được sửa đổi", " đã được sửa đổi"):
            position = summary.casefold().find(marker)
            if position >= 0:
                summary = summary[:position].rstrip(" ,;:-")
        return summary

    @classmethod
    def _normalise_whitespace(cls, value: str) -> str:
        return cls._SUMMARY_CLEANUP.sub(" ", value).strip()

    @staticmethod
    def _best_summary(summaries: Sequence[str]) -> str:
        # Source notes for later articles often omit the trích yếu.  The most
        # descriptive note for the same official document restores it faithfully.
        return max(set(summaries), key=lambda value: (len(value), value), default="")

    @staticmethod
    def _document_title(seed: _CitationSeed, fallback_summary: str) -> str:
        summary = seed.summary or fallback_summary
        return " ".join(
            component
            for component in (seed.document_type, seed.document_code, summary)
            if component
        )
