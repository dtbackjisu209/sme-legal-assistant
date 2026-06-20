from __future__ import annotations

import os
import tempfile
from collections.abc import Sequence
from pathlib import Path

from ai_legal_assistant.domain.entities.competition_submission import SubmissionRecord
from ai_legal_assistant.infrastructure.persistence.json_submission_file import JsonSubmissionFile


class JsonSubmissionCheckpoint:
    """An atomic local checkpoint; it is removed after the final ZIP is created."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> list[SubmissionRecord]:
        if not self.path.exists():
            return []
        return JsonSubmissionFile.load(self.path)

    def save(self, records: Sequence[SubmissionRecord]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=self.path.parent,
            delete=False,
        )
        try:
            with handle:
                handle.write(JsonSubmissionFile.dump(list(records)))
            os.replace(handle.name, self.path)
        except Exception:
            Path(handle.name).unlink(missing_ok=True)
            raise

    def clear(self) -> None:
        self.path.unlink(missing_ok=True)
