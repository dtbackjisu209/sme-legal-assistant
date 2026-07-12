from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch


ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ai_legal_assistant.application.dto.vector_import_dto import ImportParquetVectorStoreCommand
from ai_legal_assistant.application.dto.vector_store_dto import (
    VectorPoint,
    VectorShard,
    VectorShardManifest,
)
from ai_legal_assistant.application.use_cases.import_parquet_vector_store import (
    FINAL_INDEXING_THRESHOLD,
    ImportParquetVectorStoreUseCase,
    content_point_id,
)


class FakeShardReader:
    def __init__(self) -> None:
        self.manifest = VectorShardManifest(
            shards=[
                VectorShard(path=Path("embedded_000.parquet"), number=0, row_count=1),
                VectorShard(path=Path("embedded_001.parquet"), number=1, row_count=1),
            ],
            total_rows=2,
        )
        self.rows_by_number: dict[int, list[list[dict[str, Any]]]] = {
            1: [
                [
                    {
                        "point_id": "legacy-1",
                        "vector": [1, 2, 3],
                        "chunk_id": "chunk-1",
                        "text": "Noi dung",
                    }
                ]
            ]
        }

    def load_manifest(self, input_dir: Path) -> VectorShardManifest:
        return self.manifest

    def iter_batches(
        self,
        shard: VectorShard,
        batch_size: int,
    ) -> list[list[dict[str, Any]]]:
        return self.rows_by_number[shard.number]


class FakeVectorStore:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.upserted: list[VectorPoint] = []

    def ensure_collection(self, vector_size: int, recreate: bool = False) -> None:
        self.calls.append(f"ensure:{vector_size}:{recreate}")

    def create_payload_indexes(self) -> None:
        self.calls.append("payload_indexes")

    def upsert(self, points: list[VectorPoint]) -> None:
        self.calls.append(f"upsert:{len(points)}")
        self.upserted.extend(points)

    def set_indexing_threshold(self, threshold: int) -> None:
        self.calls.append(f"threshold:{threshold}")

    def count(self) -> int:
        self.calls.append("count")
        return 1


class FakeProgressReporter:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def report(self, message: str) -> None:
        self.messages.append(message)


def test_imports_selected_parquet_shards_into_vector_store() -> None:
    reader = FakeShardReader()
    vector_store = FakeVectorStore()
    reporter = FakeProgressReporter()
    command = ImportParquetVectorStoreCommand(
        input_dir=Path("data/embeddings/qwen3_06b"),
        collection_name="law_chunks_qwen3_06b",
        vector_size=3,
        batch_size=512,
        start_file=1,
        recreate=True,
    )

    with patch("ai_legal_assistant.application.use_cases.import_parquet_vector_store.time.sleep"):
        result = ImportParquetVectorStoreUseCase(
            vector_shard_reader=reader,
            vector_store=vector_store,
            progress_reporter=reporter,
            command=command,
        ).execute()

    assert vector_store.calls == [
        "ensure:3:True",
        "threshold:0",
        "upsert:1",
        "payload_indexes",
        f"threshold:{FINAL_INDEXING_THRESHOLD}",
        "count",
    ]
    assert result.imported_count == 1
    assert result.unique_point_count == 1
    assert result.collection_count == 1
    assert result.first_shard_number == 1
    assert result.last_shard_number == 1
    assert reporter.messages[:2] == [
        "Found 2 shards with 2 total rows.",
        "Importing shards 001-001 into 'law_chunks_qwen3_06b'.",
    ]

    [point] = vector_store.upserted
    assert point.point_id == content_point_id({"chunk_id": "chunk-1", "text": "Noi dung"})
    assert point.vector == [1.0, 2.0, 3.0]
    assert point.payload == {
        "chunk_id": "chunk-1",
        "text": "Noi dung",
        "legacy_point_id": "legacy-1",
    }


def test_can_skip_payload_indexes() -> None:
    reader = FakeShardReader()
    vector_store = FakeVectorStore()
    command = ImportParquetVectorStoreCommand(
        input_dir=Path("data/embeddings/qwen3_06b"),
        collection_name="law_chunks_qwen3_06b",
        vector_size=3,
        create_payload_indexes=False,
        start_file=1,
    )

    ImportParquetVectorStoreUseCase(
        vector_shard_reader=reader,
        vector_store=vector_store,
        command=command,
    ).execute()

    assert "payload_indexes" not in vector_store.calls
