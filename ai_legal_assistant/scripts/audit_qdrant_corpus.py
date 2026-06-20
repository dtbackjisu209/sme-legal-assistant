from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare local law chunk IDs with a Qdrant collection."
    )
    parser.add_argument(
        "--chunks-path",
        type=Path,
        default=ROOT_DIR / "data" / "processed" / "law_chunks.jsonl",
    )
    parser.add_argument(
        "--collection",
        default=os.getenv("QDRANT_COLLECTION", "law_chunks_qwen3_06b"),
    )
    parser.add_argument(
        "--qdrant-url",
        default=os.getenv("QDRANT_URL", "http://localhost:6333"),
    )
    parser.add_argument("--qdrant-api-key", default=os.getenv("QDRANT_API_KEY"))
    parser.add_argument("--batch-size", type=int, default=1_000)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_DIR / "data" / "eval" / "qdrant_corpus_audit",
    )
    return parser.parse_args()


def load_local_chunks(path: Path) -> tuple[dict[str, dict[str, Any]], list[str]]:
    chunks: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            chunk_id = str(row.get("chunk_id") or "").strip()
            text = str(row.get("text") or "").strip()
            if not chunk_id or not text:
                continue
            if chunk_id in chunks:
                duplicates.append(chunk_id)
                continue
            row["_source_line"] = line_number
            chunks[chunk_id] = row
    return chunks, duplicates


def load_qdrant_chunk_ids(
    *,
    url: str,
    api_key: str | None,
    collection: str,
    batch_size: int,
) -> tuple[dict[str, list[str]], int, int]:
    try:
        from qdrant_client import QdrantClient
    except ImportError as exc:
        raise RuntimeError("Missing dependency: qdrant-client.") from exc

    client = QdrantClient(url=url, api_key=api_key, timeout=120.0)
    points_by_chunk: dict[str, list[str]] = defaultdict(list)
    missing_payload_count = 0
    point_count = 0
    offset: Any = None

    while True:
        points, next_offset = client.scroll(
            collection_name=collection,
            limit=batch_size,
            offset=offset,
            with_payload=["chunk_id"],
            with_vectors=False,
        )
        for point in points:
            point_count += 1
            chunk_id = str((point.payload or {}).get("chunk_id") or "").strip()
            if not chunk_id:
                missing_payload_count += 1
                continue
            points_by_chunk[chunk_id].append(str(point.id))
        if next_offset is None:
            break
        offset = next_offset

    return dict(points_by_chunk), point_count, missing_payload_count


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")

    local_chunks, local_duplicates = load_local_chunks(args.chunks_path)
    qdrant_points, qdrant_point_count, missing_payload_count = load_qdrant_chunk_ids(
        url=args.qdrant_url,
        api_key=args.qdrant_api_key,
        collection=args.collection,
        batch_size=args.batch_size,
    )

    local_ids = set(local_chunks)
    qdrant_ids = set(qdrant_points)
    missing_ids = sorted(local_ids - qdrant_ids)
    extra_ids = sorted(qdrant_ids - local_ids)
    qdrant_duplicates = {
        chunk_id: point_ids
        for chunk_id, point_ids in qdrant_points.items()
        if len(point_ids) > 1
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(
        args.output_dir / "missing_in_qdrant.jsonl",
        [local_chunks[chunk_id] for chunk_id in missing_ids],
    )
    write_jsonl(
        args.output_dir / "extra_in_qdrant.jsonl",
        [{"chunk_id": chunk_id, "point_ids": qdrant_points[chunk_id]} for chunk_id in extra_ids],
    )
    write_jsonl(
        args.output_dir / "duplicate_chunk_ids_in_qdrant.jsonl",
        [
            {"chunk_id": chunk_id, "point_ids": point_ids}
            for chunk_id, point_ids in sorted(qdrant_duplicates.items())
        ],
    )

    summary = {
        "local_unique_chunk_ids": len(local_ids),
        "local_duplicate_rows": len(local_duplicates),
        "qdrant_points": qdrant_point_count,
        "qdrant_unique_chunk_ids": len(qdrant_ids),
        "qdrant_points_without_chunk_id": missing_payload_count,
        "qdrant_duplicate_chunk_ids": len(qdrant_duplicates),
        "missing_in_qdrant": len(missing_ids),
        "extra_in_qdrant": len(extra_ids),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Reports written to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
