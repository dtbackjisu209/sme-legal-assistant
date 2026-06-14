from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LawArticle:
    article_id: str
    subject_id: str | None
    subject_number: int | None
    subject_title: str | None
    topic_id: str | None
    topic_number: int | None
    topic_title: str | None
    article_anchor: str | None
    article_title: str | None
    chapter_title: str | None
    source_note_text: str | None
    related_note_text: str | None
    source_links: list[dict[str, Any]]
    source_url: str | None
    scraped_at: str | None
    text: str
    char_len: int
    word_count: int
