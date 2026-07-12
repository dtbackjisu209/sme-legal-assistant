from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class VectorPoint:
    point_id: str
    vector: list[float]
    payload: dict[str, Any]


@dataclass(frozen=True)
class VectorShard:
    path: Path
    number: int
    row_count: int


@dataclass(frozen=True)
class VectorShardManifest:
    shards: list[VectorShard]
    total_rows: int
