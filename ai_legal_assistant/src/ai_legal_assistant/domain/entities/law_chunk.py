from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LawChunk:
    chunk_id: str
    article_id: str
    article_title: str | None
    subject_id: str | None
    subject_number: int | None
    subject_title: str | None
    topic_id: str | None
    topic_number: int | None
    topic_title: str | None
    chapter_title: str | None
    source_url: str | None
    chunk_type: str
    clause_number: str | None
    point_label: str | None
    text: str
    char_len: int
    word_count: int
    ordinal: int
    parent_chunk_id: str | None
    is_subchunk: bool
    subchunk_index: int | None
    subchunk_count: int | None
    start_char: int
    end_char: int
