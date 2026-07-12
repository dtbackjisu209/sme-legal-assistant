from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ai_legal_assistant.application.dto.vector_store_dto import VectorShard, VectorShardManifest


POINT_COLUMNS = ("point_id", "vector")


class ParquetVectorShardReader:
    def load_manifest(self, input_dir: Path) -> VectorShardManifest:
        pq = _load_parquet_module()
        files = sorted(input_dir.glob("embedded_*.parquet"), key=self.shard_number)
        if not files:
            raise FileNotFoundError("No embedded_*.parquet files found.")

        numbers = [self.shard_number(path) for path in files]
        expected = list(range(numbers[0], numbers[-1] + 1))
        if numbers != expected:
            missing = sorted(set(expected) - set(numbers))
            raise ValueError(f"Missing Parquet shards: {missing}")

        shards: list[VectorShard] = []
        total_rows = 0
        for path, number in zip(files, numbers, strict=True):
            parquet_file = pq.ParquetFile(path)
            columns = set(parquet_file.schema_arrow.names)
            missing_columns = set(POINT_COLUMNS) - columns
            if missing_columns:
                raise ValueError(f"{path.name} is missing columns: {sorted(missing_columns)}")
            row_count = parquet_file.metadata.num_rows
            total_rows += row_count
            shards.append(VectorShard(path=path, number=number, row_count=row_count))

        return VectorShardManifest(shards=shards, total_rows=total_rows)

    def iter_batches(
        self,
        shard: VectorShard,
        batch_size: int,
    ) -> Iterable[list[dict[str, Any]]]:
        pq = _load_parquet_module()
        parquet_file = pq.ParquetFile(shard.path)
        for record_batch in parquet_file.iter_batches(batch_size=batch_size):
            yield record_batch.to_pylist()

    @staticmethod
    def shard_number(path: Path) -> int:
        try:
            return int(path.stem.rsplit("_", maxsplit=1)[1])
        except (IndexError, ValueError) as exc:
            raise ValueError(f"Unexpected shard filename: {path.name}") from exc


def _load_parquet_module() -> Any:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Missing dependency: pyarrow.") from exc
    return pq
