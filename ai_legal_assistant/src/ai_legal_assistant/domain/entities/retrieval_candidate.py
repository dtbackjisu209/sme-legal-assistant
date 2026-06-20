from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RetrievalCandidate:
    content_id: str
    legacy_chunk_ids: tuple[str, ...]
    text: str
    metadata: dict[str, Any]
    rrf_score: float
    matched_scopes: tuple[str, ...] = ()


@dataclass(frozen=True)
class RerankedCandidate:
    candidate: RetrievalCandidate
    relevance_score: float

