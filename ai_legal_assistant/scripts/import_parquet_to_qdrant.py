from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ai_legal_assistant.application.ports.vector_store_port import VectorPoint
from ai_legal_assistant.infrastructure.vectorstores.qdrant_vector_store import QdrantVectorStore


VECTOR_SIZE = 1024
POINT_COLUMNS = ("point_id", "vector")
POINT_ID_NAMESPACE = uuid.UUID("734a5798-fdd0-406d-842d-2b3c446a8f28")
FINAL_INDEXING_THRESHOLD = 10_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import embedded Parquet shards into Qdrant.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=ROOT_DIR / "data" / "embeddings" / "qwen3_06b",
        help="Directory containing embedded_*.parquet files.",
    )
    parser.add_argument(
        "--collection",
        default=os.getenv("QDRANT_COLLECTION", "law_chunks_qwen3_06b"),
        help="Qdrant collection name.",
    )
    parser.add_argument(
        "--qdrant-url",
        default=os.getenv("QDRANT_URL", "http://localhost:6333"),
        help="Qdrant URL.",
    )
    parser.add_argument(
        "--qdrant-api-key",
        default=os.getenv("QDRANT_API_KEY"),
        help="Optional Qdrant API key.",
    )
    parser.add_argument("--batch-size", type=int, default=512, help="Points per Qdrant upsert.")
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=120.0,
        help="HTTP timeout for each Qdrant operation.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Maximum retries for a failed Qdrant batch.",
    )
    parser.add_argument(
        "--start-file",
        type=int,
        default=0,
        help="First shard number to import, useful when resuming.",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete and recreate the collection before importing.",
    )
    parser.add_argument(
        "--no-payload-indexes",
        action="store_true",
        help="Do not create payload indexes after importing.",
    )
    return parser.parse_args()


def shard_number(path: Path) -> int:
    try:
        return int(path.stem.rsplit("_", maxsplit=1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Unexpected shard filename: {path.name}") from exc


def validate_shards(files: list[Path]) -> int:
    if not files:
        raise FileNotFoundError("No embedded_*.parquet files found.")

    numbers = [shard_number(path) for path in files]
    expected = list(range(numbers[0], numbers[-1] + 1))
    if numbers != expected:
        missing = sorted(set(expected) - set(numbers))
        raise ValueError(f"Missing Parquet shards: {missing}")

    total_rows = 0
    for path in files:
        parquet_file = pq.ParquetFile(path)
        columns = set(parquet_file.schema_arrow.names)
        missing_columns = set(POINT_COLUMNS) - columns
        if missing_columns:
            raise ValueError(f"{path.name} is missing columns: {sorted(missing_columns)}")
        total_rows += parquet_file.metadata.num_rows
    return total_rows


def build_points(rows: list[dict[str, Any]], source: Path) -> list[VectorPoint]:
    points: list[VectorPoint] = []
    for row in rows:
        vector = row.pop("vector")
        legacy_point_id = row.pop("point_id")
        if len(vector) != VECTOR_SIZE:
            raise ValueError(
                f"{source.name} contains vector dimension {len(vector)}; expected {VECTOR_SIZE}."
            )
        points.append(
            VectorPoint(
                point_id=content_point_id(row),
                vector=[float(value) for value in vector],
                payload={**row, "legacy_point_id": str(legacy_point_id)},
            )
        )
    return points


def content_point_id(payload: dict[str, Any]) -> str:
    canonical_payload = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return str(uuid.uuid5(POINT_ID_NAMESPACE, f"parquet-payload:{canonical_payload}"))


def upsert_with_retry(
    vector_store: QdrantVectorStore,
    points: list[VectorPoint],
    *,
    max_retries: int,
) -> None:
    for attempt in range(max_retries + 1):
        try:
            vector_store.upsert(points)
            return
        except Exception as exc:
            if attempt >= max_retries:
                raise
            delay = min(2 ** attempt, 30)
            print(
                f"Qdrant batch failed ({type(exc).__name__}); retrying in {delay}s "
                f"[{attempt + 1}/{max_retries}]...",
                flush=True,
            )
            time.sleep(delay)


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.timeout_seconds <= 0:
        raise ValueError("--timeout-seconds must be positive.")
    if args.max_retries < 0:
        raise ValueError("--max-retries cannot be negative.")
    if args.start_file < 0:
        raise ValueError("--start-file cannot be negative.")

    all_files = sorted(args.input_dir.glob("embedded_*.parquet"), key=shard_number)
    total_rows = validate_shards(all_files)
    files = [path for path in all_files if shard_number(path) >= args.start_file]
    if not files:
        raise ValueError(f"No shards remain at or after --start-file {args.start_file}.")

    print(f"Found {len(all_files)} shards with {total_rows:,} total rows.", flush=True)
    print(
        f"Importing shards {shard_number(files[0]):03d}-{shard_number(files[-1]):03d} "
        f"into '{args.collection}'.",
        flush=True,
    )

    vector_store = QdrantVectorStore(
        collection_name=args.collection,
        url=args.qdrant_url,
        api_key=args.qdrant_api_key,
        timeout_seconds=args.timeout_seconds,
    )
    vector_store.ensure_collection(vector_size=VECTOR_SIZE, recreate=args.recreate)
    print("Temporarily disabling HNSW indexing during bulk import...", flush=True)
    vector_store.set_indexing_threshold(0)

    started_at = time.monotonic()
    imported = 0
    generated_ids: set[str] = set()
    for file_index, path in enumerate(files, start=1):
        parquet_file = pq.ParquetFile(path)
        file_rows = 0
        for record_batch in parquet_file.iter_batches(batch_size=args.batch_size):
            rows = record_batch.to_pylist()
            points = build_points(rows, path)
            upsert_with_retry(vector_store, points, max_retries=args.max_retries)
            generated_ids.update(point.point_id for point in points)
            imported += len(points)
            file_rows += len(points)

        elapsed = max(time.monotonic() - started_at, 0.001)
        print(
            f"[{file_index:02d}/{len(files):02d}] {path.name}: {file_rows:,} points "
            f"| total={imported:,} | {imported / elapsed:.1f} points/s",
            flush=True,
        )

    if not args.no_payload_indexes:
        print("Creating payload indexes...", flush=True)
        vector_store.create_payload_indexes()

    print("Enabling HNSW indexing...", flush=True)
    vector_store.set_indexing_threshold(FINAL_INDEXING_THRESHOLD)
    collection_count = vector_store.count()
    print(
        f"Done. Processed {imported:,} rows ({len(generated_ids):,} unique points); "
        f"collection contains {collection_count:,} points.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
