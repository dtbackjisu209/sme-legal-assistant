from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from ai_legal_assistant.domain.entities.bm25_index import SearchableLawChunk


class JsonlLawChunkReader:
    def __init__(self, chunks_path: Path) -> None:
        self.chunks_path = chunks_path

    def iter_chunks(self, limit: int | None = None) -> Iterator[SearchableLawChunk]:
        if not self.chunks_path.exists():
            raise FileNotFoundError(f"Law chunks file not found: {self.chunks_path}")

        emitted = 0
        with self.chunks_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if limit is not None and emitted >= limit:
                    break
                if not line.strip():
                    continue

                row = self._parse_row(line=line, line_number=line_number)
                chunk_id = str(row.get("chunk_id") or "")
                text = str(row.get("text") or "")
                if not chunk_id:
                    raise ValueError(f"Missing chunk_id at {self.chunks_path}:{line_number}")
                if not text.strip():
                    continue

                emitted += 1
                yield SearchableLawChunk(
                    chunk_id=chunk_id,
                    text=text,
                    metadata={key: value for key, value in row.items() if key != "text"},
                )

    def _parse_row(self, line: str, line_number: int) -> dict[str, object]:
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at {self.chunks_path}:{line_number}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"Expected JSON object at {self.chunks_path}:{line_number}")
        return row
