from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import TextIO

from ai_legal_assistant.domain.entities.law_article import LawArticle
from ai_legal_assistant.domain.entities.law_chunk import LawChunk


class JsonlLawRepository:
    def __init__(self, output_dir: Path = Path("data/processed")) -> None:
        self.output_dir = output_dir
        self._article_file: TextIO | None = None
        self._chunk_file: TextIO | None = None

    @property
    def articles_path(self) -> Path:
        return self.output_dir / "law_articles.jsonl"

    @property
    def chunks_path(self) -> Path:
        return self.output_dir / "law_chunks.jsonl"

    def open(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._article_file = self.articles_path.open("w", encoding="utf-8", newline="\n")
        self._chunk_file = self.chunks_path.open("w", encoding="utf-8", newline="\n")

    def close(self) -> None:
        if self._article_file is not None:
            self._article_file.close()
            self._article_file = None
        if self._chunk_file is not None:
            self._chunk_file.close()
            self._chunk_file = None

    def save_article(self, article: LawArticle) -> None:
        self._write(self._article_file, article)

    def save_chunk(self, chunk: LawChunk) -> None:
        self._write(self._chunk_file, chunk)

    def _write(self, handle: TextIO | None, row: LawArticle | LawChunk) -> None:
        if handle is None:
            raise RuntimeError("JsonlLawRepository must be opened before writing.")
        handle.write(json.dumps(asdict(row), ensure_ascii=False, separators=(",", ":")) + "\n")
