from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from enum import Enum
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ai_legal_assistant.application.dto.retrieval_dto import RetrieveLegalContextQuery
from ai_legal_assistant.infrastructure.bootstrap.retrieval import (
    DenseRetrievalConfig,
    build_dense_retriever,
    build_expanded_dense_retriever,
)
from ai_legal_assistant.infrastructure.embeddings.qwen3_query_embedder import (
    DEFAULT_QUERY_INSTRUCTION,
)


DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embed one legal query and search Qdrant.")
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--retrieval-mode",
        choices=("auto", "baseline"),
        default="auto",
        help=(
            "auto lets the query planner decide whether expansion is needed; "
            "baseline bypasses query analysis and searches only the original query."
        ),
    )
    parser.add_argument(
        "--expand-query",
        action="store_true",
        help="Deprecated compatibility alias for --retrieval-mode auto.",
    )
    parser.add_argument("--per-query-top-k", type=int, default=20)
    parser.add_argument(
        "--planner-model",
        default=os.getenv("QUERY_PLANNER_MODEL", "Qwen/Qwen3-0.6B"),
    )
    parser.add_argument("--planner-device", default=os.getenv("QUERY_PLANNER_DEVICE"))
    parser.add_argument("--planner-max-input-tokens", type=int, default=4096)
    parser.add_argument("--planner-max-new-tokens", type=int, default=500)
    parser.add_argument("--planner-trust-remote-code", action="store_true")
    parser.add_argument("--model", default=os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL))
    parser.add_argument(
        "--embedding-provider",
        choices=("ollama", "huggingface"),
        default=os.getenv("EMBEDDING_PROVIDER", "huggingface"),
    )
    parser.add_argument("--ollama-url", default=os.getenv("OLLAMA_URL", "http://localhost:11434"))
    parser.add_argument(
        "--collection",
        default=os.getenv("QDRANT_COLLECTION", "law_chunks_qwen3_06b"),
    )
    parser.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL", "http://localhost:6333"))
    parser.add_argument("--qdrant-api-key", default=os.getenv("QDRANT_API_KEY"))
    parser.add_argument("--vector-size", type=int, default=1024)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--query-instruction",
        default=os.getenv("QUERY_EMBEDDING_INSTRUCTION", DEFAULT_QUERY_INSTRUCTION),
    )
    parser.add_argument("--no-query-instruction", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = DenseRetrievalConfig(
        model_name_or_path=args.model,
        embedding_provider=args.embedding_provider,
        ollama_url=args.ollama_url,
        collection_name=args.collection,
        qdrant_url=args.qdrant_url,
        qdrant_api_key=args.qdrant_api_key,
        vector_size=args.vector_size,
        query_instruction=None if args.no_query_instruction else args.query_instruction,
        max_length=args.max_length,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
        planner_model_name_or_path=args.planner_model,
        planner_device=args.planner_device,
        planner_max_input_tokens=args.planner_max_input_tokens,
        planner_max_new_tokens=args.planner_max_new_tokens,
        planner_trust_remote_code=args.planner_trust_remote_code,
    )
    retrieval_mode = "auto" if args.expand_query else args.retrieval_mode
    if retrieval_mode == "auto":
        retriever = build_expanded_dense_retriever(
            config,
            per_query_top_k=args.per_query_top_k,
        )
        plan = retriever.build_plan(args.query)
        hits = retriever.execute_plan(plan, top_k=args.top_k)
        output: Any = {
            "query_plan": asdict(plan),
            "hits": [asdict(hit) for hit in hits],
        }
    else:
        retriever = build_dense_retriever(config)
        hits = retriever.execute(RetrieveLegalContextQuery(query=args.query, top_k=args.top_k))
        output = [asdict(hit) for hit in hits]

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2, default=_json_default))
    return 0


def _json_default(value: Any) -> str:
    if isinstance(value, Enum):
        return value.value
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


if __name__ == "__main__":
    raise SystemExit(main())
