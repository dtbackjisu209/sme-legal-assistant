from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from ai_legal_assistant.domain.entities.competition_submission import SubmissionRecord


class SubmissionCheckpointPort(Protocol):
    """Persists completed records so a long local generation can safely resume."""

    def load(self) -> list[SubmissionRecord]:
        ...

    def save(self, records: Sequence[SubmissionRecord]) -> None:
        ...

    def clear(self) -> None:
        ...
