from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from ai_legal_assistant.application.dto.retrieval_dto import DenseSearchResult
from ai_legal_assistant.domain.entities.competition_submission import ResolvedLegalContext


class LegalCitationResolverPort(Protocol):
    def resolve(self, hits: Sequence[DenseSearchResult]) -> list[ResolvedLegalContext]:
        ...
