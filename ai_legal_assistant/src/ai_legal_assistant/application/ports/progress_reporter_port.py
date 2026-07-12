from __future__ import annotations

from typing import Protocol


class ProgressReporterPort(Protocol):
    def report(self, message: str) -> None:
        ...
