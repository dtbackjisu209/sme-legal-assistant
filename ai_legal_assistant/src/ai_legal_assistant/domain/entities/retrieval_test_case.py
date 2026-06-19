from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ai_legal_assistant.domain.entities.legal_query import LegalQuery


RelevanceField = Literal["chunk_id", "article_id"]


@dataclass(frozen=True)
class RetrievalTestCase:
    case_id: str
    query: LegalQuery
    relevance_field: RelevanceField
    relevant_ids: frozenset[str]

    def __post_init__(self) -> None:
        if not self.case_id.strip():
            raise ValueError("Retrieval test case id cannot be empty.")
        if not self.relevant_ids:
            raise ValueError(f"Retrieval test case '{self.case_id}' has no relevant ids.")
