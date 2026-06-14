from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from typing import Any, Iterable, Iterator

from ai_legal_assistant.domain.entities.law_article import LawArticle


class HuggingFacePhapdienLoader:
    def __init__(
        self,
        dataset_name: str = "tmquan/phapdien-moj-gov-vn",
        dataset_config: str = "articles",
        split: str = "train",
    ) -> None:
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.split = split

    def load_articles(self, limit: int | None = None) -> Iterator[LawArticle]:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError(
                "Missing dependency: datasets. Install it with `pip install datasets` "
                "or add it to the project dependencies."
            ) from exc

        dataset = load_dataset(
            self.dataset_name,
            self.dataset_config,
            split=self.split,
            streaming=limit is not None,
        )
        rows: Iterable[dict[str, Any]] = dataset.take(limit) if limit is not None else dataset

        for row in rows:
            article = self._make_article(row)
            if article.text:
                yield article

    def _make_article(self, row: dict[str, Any]) -> LawArticle:
        text = self._clean_text(row.get("content_text"))
        article_id = self._stable_id(
            row.get("subject_id"),
            row.get("topic_id"),
            row.get("article_anchor"),
            row.get("article_title"),
            row.get("source_url"),
        )

        return LawArticle(
            article_id=article_id,
            subject_id=row.get("subject_id"),
            subject_number=self._safe_int(row.get("subject_number")),
            subject_title=self._clean_text(row.get("subject_title")) or None,
            topic_id=row.get("topic_id"),
            topic_number=self._safe_int(row.get("topic_number")),
            topic_title=self._clean_text(row.get("topic_title")) or None,
            article_anchor=row.get("article_anchor"),
            article_title=self._clean_text(row.get("article_title")) or None,
            chapter_title=self._clean_text(row.get("chapter_title")) or None,
            source_note_text=self._clean_text(row.get("source_note_text")) or None,
            related_note_text=self._clean_text(row.get("related_note_text")) or None,
            source_links=self._safe_links(row.get("source_links")),
            source_url=row.get("source_url"),
            scraped_at=row.get("scraped_at"),
            text=text,
            char_len=len(text),
            word_count=len(re.findall(r"\S+", text)),
        )

    def _clean_text(self, value: Any) -> str:
        if value is None:
            return ""

        text = unicodedata.normalize("NFC", str(value))
        text = text.replace("\ufeff", " ").replace("\u00a0", " ")
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n[ \t]+", "\n", text)
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _safe_int(self, value: Any) -> int | None:
        if value is None or value == "":
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _safe_links(self, value: Any) -> list[dict[str, Any]]:
        if value is None:
            return []
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                return []
            return [item for item in parsed if isinstance(item, dict)] if isinstance(parsed, list) else []
        return []

    def _stable_id(self, *parts: Any, length: int = 20) -> str:
        raw = "|".join("" if part is None else str(part) for part in parts)
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:length]
