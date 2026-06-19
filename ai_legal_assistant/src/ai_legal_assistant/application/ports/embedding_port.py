from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from ai_legal_assistant.domain.entities.legal_query import LegalQuery


class QueryEmbeddingPort(Protocol):
    def embed_query(self, query: LegalQuery) -> Sequence[float]:
        ...
