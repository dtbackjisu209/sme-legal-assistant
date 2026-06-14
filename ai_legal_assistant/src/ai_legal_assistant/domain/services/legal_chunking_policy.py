from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Iterator

from ai_legal_assistant.domain.entities.law_article import LawArticle
from ai_legal_assistant.domain.entities.law_chunk import LawChunk


@dataclass(frozen=True)
class LegalSegment:
    chunk_type: str
    clause_number: str | None
    point_label: str | None
    text: str


@dataclass(frozen=True)
class TextSpan:
    start: int
    end: int


class LegalChunkingPolicy:
    _CLAUSE_RE = re.compile(r"(?m)^(?P<num>\d+)\.\s+")
    _POINT_RE = re.compile("(?m)(?:^|(?<=[\\n;:]))\\s*(?P<label>[a-z\\u0111])\\)\\s+", re.IGNORECASE)

    def __init__(self, max_chars: int = 1800, overlap_chars: int = 250, min_chunk_chars: int = 300) -> None:
        if max_chars <= 0:
            raise ValueError("max_chars must be positive.")
        if overlap_chars < 0:
            raise ValueError("overlap_chars cannot be negative.")
        if overlap_chars >= max_chars:
            raise ValueError("overlap_chars must be smaller than max_chars.")
        if min_chunk_chars < 0:
            raise ValueError("min_chunk_chars cannot be negative.")

        self.max_chars = max_chars
        self.overlap_chars = overlap_chars
        self.min_chunk_chars = min_chunk_chars

    def chunk(self, article: LawArticle) -> Iterator[LawChunk]:
        base = {
            "article_id": article.article_id,
            "article_title": article.article_title,
            "subject_id": article.subject_id,
            "subject_number": article.subject_number,
            "subject_title": article.subject_title,
            "topic_id": article.topic_id,
            "topic_number": article.topic_number,
            "topic_title": article.topic_title,
            "chapter_title": article.chapter_title,
            "source_url": article.source_url,
        }

        ordinal = 0
        for segment in self._iter_legal_segments(article.text):
            chunks = self._make_chunks_for_segment(base, segment)
            for chunk in chunks:
                ordinal += 1
                yield self._with_ordinal(chunk, ordinal)

    def _iter_legal_segments(self, article_text: str) -> Iterator[LegalSegment]:
        for clause_number, clause_text in self._split_by_marker(article_text, self._CLAUSE_RE, "num"):
            point_parts = self._split_by_marker(clause_text, self._POINT_RE, "label")

            if clause_number is None:
                for _, text in point_parts:
                    yield LegalSegment("article", None, None, text)
                continue

            if len(point_parts) == 1 and point_parts[0][0] is None:
                yield LegalSegment("clause", clause_number, None, clause_text)
                continue

            for point_label, point_text in point_parts:
                if point_label is None:
                    yield LegalSegment("clause", clause_number, None, point_text)
                else:
                    yield LegalSegment("point", clause_number, point_label.lower(), point_text)

    def _make_chunks_for_segment(self, base: dict[str, Any], segment: LegalSegment) -> list[LawChunk]:
        parent_chunk_id = self._segment_id(base, segment)
        text = segment.text.strip()
        if len(text) <= self.max_chars:
            return [
                self._make_chunk(
                    base=base,
                    chunk_id=parent_chunk_id,
                    chunk_type=segment.chunk_type,
                    clause_number=segment.clause_number,
                    point_label=segment.point_label,
                    text=text,
                    ordinal=0,
                    parent_chunk_id=None,
                    is_subchunk=False,
                    subchunk_index=None,
                    subchunk_count=None,
                    start_char=0,
                    end_char=len(text),
                )
            ]

        spans = self._split_long_text(text)
        part_type = f"{segment.chunk_type}_part"
        return [
            self._make_chunk(
                base=base,
                chunk_id=self._stable_id(parent_chunk_id, index, span.start, span.end),
                chunk_type=part_type,
                clause_number=segment.clause_number,
                point_label=segment.point_label,
                text=text[span.start : span.end],
                ordinal=0,
                parent_chunk_id=parent_chunk_id,
                is_subchunk=True,
                subchunk_index=index,
                subchunk_count=len(spans),
                start_char=span.start,
                end_char=span.end,
            )
            for index, span in enumerate(spans, start=1)
        ]

    def _split_by_marker(
        self,
        text: str,
        marker_re: re.Pattern[str],
        group_name: str,
    ) -> list[tuple[str | None, str]]:
        matches = list(marker_re.finditer(text))
        if not matches:
            return [(None, text.strip())] if text.strip() else []

        parts: list[tuple[str | None, str]] = []
        intro = text[: matches[0].start()].strip()
        if intro:
            parts.append((None, intro))

        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            parts.append((match.group(group_name), text[match.start() : end].strip()))

        return [(label, part) for label, part in parts if part]

    def _split_long_text(self, text: str) -> list[TextSpan]:
        spans: list[TextSpan] = []
        text_len = len(text)
        start = 0

        while start < text_len:
            raw_end = min(start + self.max_chars, text_len)
            end = text_len if raw_end == text_len else self._find_breakpoint(text, start, raw_end)
            start, end = self._trim_span(text, start, end)

            if end <= start:
                end = min(start + self.max_chars, text_len)

            spans.append(TextSpan(start, end))

            if end >= text_len:
                break

            next_start = max(0, end - self.overlap_chars)
            if next_start <= start:
                next_start = end
            start = next_start

        return self._merge_short_tail(text, spans)

    def _find_breakpoint(self, text: str, start: int, raw_end: int) -> int:
        min_end = min(raw_end, start + max(self.min_chunk_chars, self.max_chars // 2))
        break_specs = [
            ("\n\n", 2),
            ("\n", 1),
            (". ", 2),
            ("; ", 2),
            (": ", 2),
            (", ", 2),
            (" ", 1),
        ]

        for marker, offset in break_specs:
            index = text.rfind(marker, min_end, raw_end)
            if index != -1:
                return index + offset

        return raw_end

    def _trim_span(self, text: str, start: int, end: int) -> tuple[int, int]:
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        return start, end

    def _merge_short_tail(self, text: str, spans: list[TextSpan]) -> list[TextSpan]:
        if len(spans) < 2 or self.min_chunk_chars == 0:
            return spans

        tail = spans[-1]
        if tail.end - tail.start >= self.min_chunk_chars:
            return spans

        previous = spans[-2]
        merged_start, merged_end = self._trim_span(text, previous.start, tail.end)
        return [*spans[:-2], TextSpan(merged_start, merged_end)]

    def _make_chunk(
        self,
        base: dict[str, Any],
        chunk_id: str,
        chunk_type: str,
        clause_number: str | None,
        point_label: str | None,
        text: str,
        ordinal: int,
        parent_chunk_id: str | None,
        is_subchunk: bool,
        subchunk_index: int | None,
        subchunk_count: int | None,
        start_char: int,
        end_char: int,
    ) -> LawChunk:
        return LawChunk(
            chunk_id=chunk_id,
            chunk_type=chunk_type,
            clause_number=clause_number,
            point_label=point_label,
            text=text,
            char_len=len(text),
            word_count=len(re.findall(r"\S+", text)),
            ordinal=ordinal,
            parent_chunk_id=parent_chunk_id,
            is_subchunk=is_subchunk,
            subchunk_index=subchunk_index,
            subchunk_count=subchunk_count,
            start_char=start_char,
            end_char=end_char,
            **base,
        )

    def _with_ordinal(self, chunk: LawChunk, ordinal: int) -> LawChunk:
        return LawChunk(
            chunk_id=chunk.chunk_id,
            article_id=chunk.article_id,
            article_title=chunk.article_title,
            subject_id=chunk.subject_id,
            subject_number=chunk.subject_number,
            subject_title=chunk.subject_title,
            topic_id=chunk.topic_id,
            topic_number=chunk.topic_number,
            topic_title=chunk.topic_title,
            chapter_title=chunk.chapter_title,
            source_url=chunk.source_url,
            chunk_type=chunk.chunk_type,
            clause_number=chunk.clause_number,
            point_label=chunk.point_label,
            text=chunk.text,
            char_len=chunk.char_len,
            word_count=chunk.word_count,
            ordinal=ordinal,
            parent_chunk_id=chunk.parent_chunk_id,
            is_subchunk=chunk.is_subchunk,
            subchunk_index=chunk.subchunk_index,
            subchunk_count=chunk.subchunk_count,
            start_char=chunk.start_char,
            end_char=chunk.end_char,
        )

    def _segment_id(self, base: dict[str, Any], segment: LegalSegment) -> str:
        return self._stable_id(base["article_id"], segment.chunk_type, segment.clause_number, segment.point_label)

    def _stable_id(self, *parts: Any, length: int = 20) -> str:
        raw = "|".join("" if part is None else str(part) for part in parts)
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:length]
