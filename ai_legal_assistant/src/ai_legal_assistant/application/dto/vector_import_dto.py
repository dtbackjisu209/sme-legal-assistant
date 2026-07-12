from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ImportParquetVectorStoreCommand:
    input_dir: Path
    collection_name: str
    vector_size: int = 1024
    batch_size: int = 512
    max_retries: int = 5
    start_file: int = 0
    recreate: bool = False
    create_payload_indexes: bool = True


@dataclass(frozen=True)
class ImportParquetVectorStoreResult:
    imported_count: int
    unique_point_count: int
    collection_count: int
    shard_count: int
    total_rows: int
    first_shard_number: int
    last_shard_number: int
    collection_name: str

    def __str__(self) -> str:
        return (
            f"Done. Processed {self.imported_count:,} rows "
            f"({self.unique_point_count:,} unique points); "
            f"collection contains {self.collection_count:,} points."
        )
