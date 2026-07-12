from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ai_legal_assistant.application.dto.vector_import_dto import ImportParquetVectorStoreCommand
from ai_legal_assistant.infrastructure.bootstrap.container import build_import_vector_store_container


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


def main() -> int:
    args = parse_args()
    command = ImportParquetVectorStoreCommand(
        input_dir=args.input_dir,
        collection_name=args.collection,
        batch_size=args.batch_size,
        max_retries=args.max_retries,
        start_file=args.start_file,
        recreate=args.recreate,
        create_payload_indexes=not args.no_payload_indexes,
    )
    container = build_import_vector_store_container(
        command,
        qdrant_url=args.qdrant_url,
        qdrant_api_key=args.qdrant_api_key,
        timeout_seconds=args.timeout_seconds,
    )
    result = container.import_vector_store_use_case.execute()
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
