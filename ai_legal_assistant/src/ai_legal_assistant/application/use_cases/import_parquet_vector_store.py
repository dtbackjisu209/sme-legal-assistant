from __future__ import annotations

import json
import time
import uuid
from typing import Any

from ai_legal_assistant.application.dto.vector_import_dto import (
    ImportParquetVectorStoreCommand,
    ImportParquetVectorStoreResult,
)
from ai_legal_assistant.application.dto.vector_store_dto import VectorPoint
from ai_legal_assistant.application.ports.progress_reporter_port import ProgressReporterPort
from ai_legal_assistant.application.ports.vector_shard_reader_port import VectorShardReaderPort
from ai_legal_assistant.application.ports.vector_store_port import VectorStoreImportPort


POINT_ID_NAMESPACE = uuid.UUID("734a5798-fdd0-406d-842d-2b3c446a8f28")
FINAL_INDEXING_THRESHOLD = 10_000


class _NullProgressReporter:
    def report(self, message: str) -> None:
        return None


class ImportParquetVectorStoreUseCase:
    def __init__(
        self,
        vector_shard_reader: VectorShardReaderPort,
        vector_store: VectorStoreImportPort,
        progress_reporter: ProgressReporterPort | None = None,
        command: ImportParquetVectorStoreCommand | None = None,
    ) -> None:
        self.vector_shard_reader = vector_shard_reader
        self.vector_store = vector_store
        self.progress_reporter = progress_reporter or _NullProgressReporter()
        self.command = command

    def execute(
        self,
        command: ImportParquetVectorStoreCommand | None = None,
    ) -> ImportParquetVectorStoreResult:
        effective_command = command or self.command
        if effective_command is None:
            raise ValueError("ImportParquetVectorStoreCommand is required.")

        self.validate_command(effective_command)
        manifest = self.vector_shard_reader.load_manifest(effective_command.input_dir)
        shards = [
            shard
            for shard in manifest.shards
            if shard.number >= effective_command.start_file
        ]
        if not shards:
            raise ValueError(
                f"No shards remain at or after --start-file {effective_command.start_file}."
            )

        self.progress_reporter.report(
            f"Found {len(manifest.shards)} shards with {manifest.total_rows:,} total rows."
        )
        self.progress_reporter.report(
            f"Importing shards {shards[0].number:03d}-{shards[-1].number:03d} "
            f"into '{effective_command.collection_name}'."
        )

        self.vector_store.ensure_collection(
            vector_size=effective_command.vector_size,
            recreate=effective_command.recreate,
        )
        self.progress_reporter.report(
            "Temporarily disabling HNSW indexing during bulk import..."
        )
        self.vector_store.set_indexing_threshold(0)

        started_at = time.monotonic()
        imported = 0
        generated_ids: set[str] = set()
        for file_index, shard in enumerate(shards, start=1):
            file_rows = 0
            for rows in self.vector_shard_reader.iter_batches(
                shard,
                effective_command.batch_size,
            ):
                points = self._build_points(
                    rows,
                    source_name=shard.path.name,
                    vector_size=effective_command.vector_size,
                )
                self._upsert_with_retry(
                    points,
                    max_retries=effective_command.max_retries,
                )
                generated_ids.update(point.point_id for point in points)
                imported += len(points)
                file_rows += len(points)

            elapsed = max(time.monotonic() - started_at, 0.001)
            self.progress_reporter.report(
                f"[{file_index:02d}/{len(shards):02d}] {shard.path.name}: "
                f"{file_rows:,} points | total={imported:,} | "
                f"{imported / elapsed:.1f} points/s"
            )

        if effective_command.create_payload_indexes:
            self.progress_reporter.report("Creating payload indexes...")
            self.vector_store.create_payload_indexes()

        self.progress_reporter.report("Enabling HNSW indexing...")
        self.vector_store.set_indexing_threshold(FINAL_INDEXING_THRESHOLD)
        collection_count = self.vector_store.count()

        return ImportParquetVectorStoreResult(
            imported_count=imported,
            unique_point_count=len(generated_ids),
            collection_count=collection_count,
            shard_count=len(shards),
            total_rows=manifest.total_rows,
            first_shard_number=shards[0].number,
            last_shard_number=shards[-1].number,
            collection_name=effective_command.collection_name,
        )

    def _upsert_with_retry(
        self,
        points: list[VectorPoint],
        *,
        max_retries: int,
    ) -> None:
        for attempt in range(max_retries + 1):
            try:
                self.vector_store.upsert(points)
                return
            except Exception as exc:
                if attempt >= max_retries:
                    raise
                delay = min(2 ** attempt, 30)
                self.progress_reporter.report(
                    f"Vector store batch failed ({type(exc).__name__}); "
                    f"retrying in {delay}s [{attempt + 1}/{max_retries}]..."
                )
                time.sleep(delay)

    @staticmethod
    def _build_points(
        rows: list[dict[str, Any]],
        *,
        source_name: str,
        vector_size: int,
    ) -> list[VectorPoint]:
        points: list[VectorPoint] = []
        for row in rows:
            vector = row.pop("vector")
            legacy_point_id = row.pop("point_id")
            if len(vector) != vector_size:
                raise ValueError(
                    f"{source_name} contains vector dimension {len(vector)}; "
                    f"expected {vector_size}."
                )
            points.append(
                VectorPoint(
                    point_id=content_point_id(row),
                    vector=[float(value) for value in vector],
                    payload={**row, "legacy_point_id": str(legacy_point_id)},
                )
            )
        return points

    @staticmethod
    def validate_command(command: ImportParquetVectorStoreCommand) -> None:
        if command.batch_size <= 0:
            raise ValueError("--batch-size must be positive.")
        if command.vector_size <= 0:
            raise ValueError("vector_size must be positive.")
        if command.max_retries < 0:
            raise ValueError("--max-retries cannot be negative.")
        if command.start_file < 0:
            raise ValueError("--start-file cannot be negative.")


def content_point_id(payload: dict[str, Any]) -> str:
    canonical_payload = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return str(uuid.uuid5(POINT_ID_NAMESPACE, f"parquet-payload:{canonical_payload}"))
