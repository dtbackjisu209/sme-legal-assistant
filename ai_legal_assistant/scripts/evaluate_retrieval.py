from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ai_legal_assistant.application.use_cases.evaluate_retrieval import EvaluateRetrievalUseCase
from ai_legal_assistant.domain.services.retrieval_metrics import RetrievalMetricsCalculator
from ai_legal_assistant.infrastructure.bootstrap.retrieval import (
    DenseRetrievalConfig,
    build_dense_retriever,
    build_expanded_retriever,
)
from ai_legal_assistant.infrastructure.embeddings.qwen3_query_embedder import (
    DEFAULT_QUERY_INSTRUCTION,
)
from ai_legal_assistant.infrastructure.persistence.jsonl_retrieval_testset import (
    JsonlRetrievalTestset,
)


DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate legal retrieval with Recall/MRR.")
    parser.add_argument(
        "--testset",
        type=Path,
        default=ROOT_DIR / "data" / "eval" / "retrieval_testset.jsonl",
    )
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
    parser.add_argument("--device", default=None, help="For example: cuda, cuda:0, or cpu.")
    parser.add_argument("--cutoffs", default="1,3,5,10,20")
    parser.add_argument("--expand-query", action="store_true")
    parser.add_argument("--per-query-top-k", type=int, default=20)
    parser.add_argument(
        "--bm25-index",
        type=Path,
        default=ROOT_DIR / "data" / "indexes" / "bm25_index.pkl",
    )
    parser.add_argument("--bm25-weight", type=float, default=0.7)
    parser.add_argument(
        "--use-bm25",
        action="store_true",
        help="Use BM25 alongside dense retrieval when --expand-query is enabled.",
    )
    parser.add_argument("--use-reranker", action="store_true")
    parser.add_argument(
        "--reranker-model",
        default=os.getenv("RERANKER_MODEL", DEFAULT_RERANKER_MODEL),
    )
    parser.add_argument("--reranker-device", default=os.getenv("RERANKER_DEVICE"))
    parser.add_argument("--reranker-batch-size", type=int, default=8)
    parser.add_argument("--reranker-max-length", type=int, default=512)
    parser.add_argument("--oversample-factor", type=int, default=5)
    parser.add_argument("--candidate-pool-size", type=int, default=50)
    parser.add_argument("--scope-relevance-margin", type=float, default=0.15)
    parser.add_argument(
        "--planner-model",
        default=os.getenv("QUERY_PLANNER_MODEL", "Qwen/Qwen3-0.6B"),
    )
    parser.add_argument("--planner-device", default=os.getenv("QUERY_PLANNER_DEVICE"))
    parser.add_argument("--planner-max-input-tokens", type=int, default=4096)
    parser.add_argument("--planner-max-new-tokens", type=int, default=500)
    parser.add_argument("--planner-trust-remote-code", action="store_true")
    parser.add_argument(
        "--query-instruction",
        default=os.getenv("QUERY_EMBEDDING_INSTRUCTION", DEFAULT_QUERY_INSTRUCTION),
    )
    parser.add_argument("--no-query-instruction", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def parse_cutoffs(value: str) -> tuple[int, ...]:
    try:
        cutoffs = tuple(sorted({int(item.strip()) for item in value.split(",") if item.strip()}))
    except ValueError as exc:
        raise ValueError("--cutoffs must be a comma-separated list of integers.") from exc
    if not cutoffs or any(cutoff <= 0 for cutoff in cutoffs):
        raise ValueError("--cutoffs must contain positive integers.")
    return cutoffs


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
        bm25_index_path=args.bm25_index if args.use_bm25 else None,
        bm25_weight=args.bm25_weight,
        reranker_model_name_or_path=(args.reranker_model if args.use_reranker else None),
        reranker_device=args.reranker_device,
        reranker_batch_size=args.reranker_batch_size,
        reranker_max_length=args.reranker_max_length,
        oversample_factor=args.oversample_factor,
        candidate_pool_size=args.candidate_pool_size,
        scope_relevance_margin=args.scope_relevance_margin,
    )
    if args.expand_query:
        retriever = build_expanded_retriever(
            config,
            per_query_top_k=args.per_query_top_k,
        )
    else:
        retriever = build_dense_retriever(config)
    evaluator = EvaluateRetrievalUseCase(
        retriever=retriever,
        testset=JsonlRetrievalTestset(args.testset),
        metrics=RetrievalMetricsCalculator(),
    )
    result = evaluator.execute(cutoffs=parse_cutoffs(args.cutoffs))
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
