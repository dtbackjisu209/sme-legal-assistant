from __future__ import annotations

from typing import Iterable, Protocol

from ai_legal_assistant.domain.entities.law_article import LawArticle


class DatasetLoaderPort(Protocol):
    def load_articles(self, limit: int | None = None) -> Iterable[LawArticle]:
        ...
