from __future__ import annotations

from typing import Protocol

from ai_legal_assistant.domain.entities.competition_submission import CompetitionQuestion


class CompetitionQuestionSourcePort(Protocol):
    def load(self) -> list[CompetitionQuestion]:
        ...
