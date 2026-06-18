from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class VectorPoint:
    point_id: str
    vector: list[float]
    payload: dict[str, Any]


class VectorStorePort(Protocol):
    def ensure_collection(self, vector_size: int, recreate: bool = False) -> None:
        ...

    def create_payload_indexes(self) -> None:
        ...

    def upsert(self, points: Sequence[VectorPoint]) -> None:
        ...
