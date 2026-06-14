from __future__ import annotations

from pathlib import Path
from typing import Protocol

from ai_legal_assistant.domain.entities.law_article import LawArticle
from ai_legal_assistant.domain.entities.law_chunk import LawChunk


class LawRepositoryPort(Protocol):
    @property
    def articles_path(self) -> Path:
        ...

    @property
    def chunks_path(self) -> Path:
        ...

    def open(self) -> None:
        ...

    def close(self) -> None:
        ...

    def save_article(self, article: LawArticle) -> None:
        ...

    def save_chunk(self, chunk: LawChunk) -> None:
        ...
