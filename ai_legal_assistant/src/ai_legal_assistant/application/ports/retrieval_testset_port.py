from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from ai_legal_assistant.domain.entities.retrieval_test_case import RetrievalTestCase


class RetrievalTestsetPort(Protocol):
    def load(self) -> Iterable[RetrievalTestCase]:
        ...
