from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ai_legal_assistant.domain.entities.legal_query import LegalQuery
from ai_legal_assistant.domain.entities.retrieval_test_case import RetrievalTestCase


class JsonlRetrievalTestset:
    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> Iterator[RetrievalTestCase]:
        if not self.path.exists():
            raise FileNotFoundError(f"Retrieval testset not found: {self.path}")

        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = self._parse(line, line_number)
                yield self._to_case(row, line_number)

    def _parse(self, line: str, line_number: int) -> dict[str, Any]:
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at {self.path}:{line_number}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"Expected JSON object at {self.path}:{line_number}")
        return row

    def _to_case(self, row: dict[str, Any], line_number: int) -> RetrievalTestCase:
        chunk_ids = self._clean_ids(row.get("relevant_chunk_ids"))
        article_ids = self._clean_ids(row.get("relevant_article_ids"))
        if bool(chunk_ids) == bool(article_ids):
            raise ValueError(
                f"Exactly one of relevant_chunk_ids or relevant_article_ids is required "
                f"at {self.path}:{line_number}"
            )

        relevance_field = "chunk_id" if chunk_ids else "article_id"
        relevant_ids = chunk_ids or article_ids
        return RetrievalTestCase(
            case_id=str(row.get("id") or f"case-{line_number}"),
            query=LegalQuery(str(row.get("query") or "")),
            relevance_field=relevance_field,
            relevant_ids=frozenset(relevant_ids),
        )

    @staticmethod
    def _clean_ids(value: Any) -> set[str]:
        if value is None:
            return set()
        if not isinstance(value, list):
            raise ValueError("Relevant ids must be a JSON array.")
        return {str(item).strip() for item in value if str(item).strip()}
