from __future__ import annotations

from ai_legal_assistant.domain.entities.law_chunk import LawChunk


class LegalEmbeddingTextPolicy:
    def build(self, chunk: LawChunk) -> str:
        parts = [
            self._line("Chu de", chunk.subject_title),
            self._line("De muc", chunk.topic_title),
            self._line("Chuong", chunk.chapter_title),
            self._line("Dieu", chunk.article_title),
            self._line("Khoan", chunk.clause_number),
            self._line("Diem", chunk.point_label),
            "",
            "Noi dung:",
            chunk.text,
        ]
        return "\n".join(part for part in parts if part is not None).strip()

    def _line(self, label: str, value: str | None) -> str | None:
        if not value:
            return None
        return f"{label}: {value}"
