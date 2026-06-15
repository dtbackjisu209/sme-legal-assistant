from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ai_legal_assistant.domain.entities.bm25_index import BM25Index


@dataclass(frozen=True)
class BM25IndexPayload:
    index: BM25Index
    tokenizer_config: dict[str, Any] = field(default_factory=dict)


class PickleBM25IndexStore:
    def __init__(self, index_path: Path, tokenizer_config: dict[str, Any] | None = None) -> None:
        self.index_path = index_path
        self.tokenizer_config = tokenizer_config or {}

    def save(self, index: BM25Index) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        payload = BM25IndexPayload(index=index, tokenizer_config=dict(self.tokenizer_config))
        with self.index_path.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self) -> BM25Index:
        return self.load_payload().index

    def load_payload(self) -> BM25IndexPayload:
        if not self.index_path.exists():
            raise FileNotFoundError(f"BM25 index file not found: {self.index_path}")

        with self.index_path.open("rb") as handle:
            payload = pickle.load(handle)

        if isinstance(payload, BM25IndexPayload):
            return payload
        if isinstance(payload, BM25Index):
            return BM25IndexPayload(index=payload)

        if not isinstance(payload, BM25Index):
            raise TypeError(f"Unexpected BM25 index payload in {self.index_path}")
        return BM25IndexPayload(index=payload)
