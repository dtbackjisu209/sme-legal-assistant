from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any, Protocol

from ai_legal_assistant.application.dto.vector_store_dto import VectorShard, VectorShardManifest


class VectorShardReaderPort(Protocol):
    def load_manifest(self, input_dir: Path) -> VectorShardManifest:
        ...

    def iter_batches(
        self,
        shard: VectorShard,
        batch_size: int,
    ) -> Iterable[list[dict[str, Any]]]:
        ...
