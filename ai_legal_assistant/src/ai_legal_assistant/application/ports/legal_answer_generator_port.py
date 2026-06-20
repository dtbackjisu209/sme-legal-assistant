from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from ai_legal_assistant.domain.entities.competition_submission import (
    CompetitionQuestion,
    ResolvedLegalContext,
)


class LegalAnswerGeneratorPort(Protocol):
    def generate(
        self,
        *,
        question: CompetitionQuestion,
        contexts: Sequence[ResolvedLegalContext],
    ) -> str:
        ...
