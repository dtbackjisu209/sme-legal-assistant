from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from ai_legal_assistant.application.dto.submission_dto import SubmissionArtifact
from ai_legal_assistant.domain.entities.competition_submission import SubmissionRecord


class SubmissionArtifactPort(Protocol):
    def write(
        self,
        *,
        records: Sequence[SubmissionRecord],
        output_dir: Path,
    ) -> SubmissionArtifact:
        ...
