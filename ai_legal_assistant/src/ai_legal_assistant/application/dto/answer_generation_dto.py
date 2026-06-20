from __future__ import annotations

from dataclasses import dataclass

from ai_legal_assistant.domain.entities.competition_submission import (
    CompetitionQuestion,
    ResolvedLegalContext,
)


@dataclass(frozen=True)
class GenerateGroundedAnswerRequest:
    """One legal answer request prepared after retrieval and citation resolution."""

    question: CompetitionQuestion
    contexts: tuple[ResolvedLegalContext, ...]
