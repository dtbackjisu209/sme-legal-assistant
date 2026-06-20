from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ai_legal_assistant.application.dto.submission_dto import GenerateSubmissionCommand
from ai_legal_assistant.application.use_cases.generate_competition_submission import (
    GenerateCompetitionSubmissionUseCase,
)
from ai_legal_assistant.domain.services.submission_validator import SubmissionValidator
from ai_legal_assistant.infrastructure.bootstrap.retrieval import (
    DenseRetrievalConfig,
    build_dense_retriever,
    build_expanded_retriever,
)
from ai_legal_assistant.infrastructure.citations.jsonl_legal_citation_resolver import (
    JsonlLegalCitationResolver,
)
from ai_legal_assistant.infrastructure.embeddings.qwen3_query_embedder import (
    DEFAULT_QUERY_INSTRUCTION,
)
from ai_legal_assistant.infrastructure.llm.grounded_legal_answer_generator import (
    GroundedLegalAnswerGenerator,
    GroundedLegalAnswerGeneratorConfig,
)
from ai_legal_assistant.infrastructure.llm.huggingface_causal_llm import (
    HuggingFaceCausalLLM,
    HuggingFaceCausalLLMConfig,
)
from ai_legal_assistant.infrastructure.persistence.json_competition_question_source import (
    JsonCompetitionQuestionSource,
)
from ai_legal_assistant.infrastructure.persistence.json_submission_checkpoint import (
    JsonSubmissionCheckpoint,
)
from ai_legal_assistant.infrastructure.persistence.submission_artifact_writer import (
    SubmissionArtifactWriter,
)


DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_PLANNER_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_ANSWER_MODEL = "Qwen/Qwen3-8B"
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
DEFAULT_QUESTIONS_PATH = ROOT_DIR / "data" / "raw" / "R2AIStage1DATA (1).json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a valid results.json and flat submission.zip for the legal QA competition."
    )
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS_PATH)
    parser.add_argument(
        "--articles",
        type=Path,
        default=ROOT_DIR / "data" / "processed" / "law_articles.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_DIR / "data" / "submissions" / "private_candidate",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        help="Optional partial-results file. Defaults to partial_results.json in --output-dir.",
    )
    parser.add_argument("--retrieval-top-k", type=int, default=8)
    parser.add_argument(
        "--retrieval-mode",
        choices=("auto", "baseline"),
        default="auto",
        help="auto uses query planning and hybrid retrieval; baseline uses dense retrieval only.",
    )

    parser.add_argument("--model", default=os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL))
    parser.add_argument(
        "--embedding-provider",
        choices=("ollama", "huggingface"),
        default=os.getenv("EMBEDDING_PROVIDER", "huggingface"),
    )
    parser.add_argument("--ollama-url", default=os.getenv("OLLAMA_URL", "http://localhost:11434"))
    parser.add_argument("--collection", default=os.getenv("QDRANT_COLLECTION", "law_chunks_qwen3_06b"))
    parser.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL", "http://localhost:6333"))
    parser.add_argument("--qdrant-api-key", default=os.getenv("QDRANT_API_KEY"))
    parser.add_argument("--vector-size", type=int, default=1024)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--embedding-device", default=os.getenv("EMBEDDING_DEVICE"))
    parser.add_argument("--embedding-trust-remote-code", action="store_true")
    parser.add_argument(
        "--query-instruction",
        default=os.getenv("QUERY_EMBEDDING_INSTRUCTION", DEFAULT_QUERY_INSTRUCTION),
    )
    parser.add_argument("--no-query-instruction", action="store_true")

    parser.add_argument("--planner-model", default=os.getenv("QUERY_PLANNER_MODEL", DEFAULT_PLANNER_MODEL))
    parser.add_argument("--planner-device", default=os.getenv("QUERY_PLANNER_DEVICE"))
    parser.add_argument("--planner-max-input-tokens", type=int, default=4096)
    parser.add_argument("--planner-max-new-tokens", type=int, default=500)
    parser.add_argument("--planner-trust-remote-code", action="store_true")
    parser.add_argument("--per-query-top-k", type=int, default=20)

    parser.add_argument(
        "--bm25-index",
        type=Path,
        default=ROOT_DIR / "data" / "indexes" / "bm25_index.pkl",
    )
    parser.add_argument("--disable-bm25", action="store_true")
    parser.add_argument("--bm25-weight", type=float, default=0.7)
    parser.add_argument("--reranker-model", default=os.getenv("RERANKER_MODEL", DEFAULT_RERANKER_MODEL))
    parser.add_argument("--reranker-device", default=os.getenv("RERANKER_DEVICE"))
    parser.add_argument("--reranker-batch-size", type=int, default=8)
    parser.add_argument("--reranker-max-length", type=int, default=512)
    parser.add_argument("--reranker-trust-remote-code", action="store_true")
    parser.add_argument("--disable-reranker", action="store_true")
    parser.add_argument("--oversample-factor", type=int, default=5)
    parser.add_argument("--candidate-pool-size", type=int, default=50)
    parser.add_argument("--scope-candidates-per-branch", type=int, default=5)
    parser.add_argument("--max-chunks-per-article", type=int, default=2)
    parser.add_argument("--ambiguous-max-chunks-per-article", type=int, default=1)
    parser.add_argument("--scope-relevance-margin", type=float, default=0.15)

    parser.add_argument("--answer-model", default=os.getenv("ANSWER_MODEL", DEFAULT_ANSWER_MODEL))
    parser.add_argument("--answer-device", default=os.getenv("ANSWER_DEVICE"))
    parser.add_argument(
        "--answer-load-in-4bit",
        action="store_true",
        help="Load the answer model with NF4 quantization; requires CUDA and bitsandbytes.",
    )
    parser.add_argument("--answer-max-input-tokens", type=int, default=16_384)
    parser.add_argument("--answer-max-new-tokens", type=int, default=700)
    parser.add_argument("--answer-trust-remote-code", action="store_true")
    parser.add_argument("--max-context-characters", type=int, default=16_000)
    parser.add_argument("--max-characters-per-context", type=int, default=3_000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    retriever_config = DenseRetrievalConfig(
        model_name_or_path=args.model,
        embedding_provider=args.embedding_provider,
        ollama_url=args.ollama_url,
        collection_name=args.collection,
        qdrant_url=args.qdrant_url,
        qdrant_api_key=args.qdrant_api_key,
        vector_size=args.vector_size,
        query_instruction=None if args.no_query_instruction else args.query_instruction,
        max_length=args.max_length,
        device=args.embedding_device,
        trust_remote_code=args.embedding_trust_remote_code,
        planner_model_name_or_path=args.planner_model,
        planner_device=args.planner_device,
        planner_max_input_tokens=args.planner_max_input_tokens,
        planner_max_new_tokens=args.planner_max_new_tokens,
        planner_trust_remote_code=args.planner_trust_remote_code,
        bm25_index_path=None if args.disable_bm25 else args.bm25_index,
        bm25_weight=args.bm25_weight,
        reranker_model_name_or_path=(None if args.disable_reranker else args.reranker_model),
        reranker_device=args.reranker_device,
        reranker_batch_size=args.reranker_batch_size,
        reranker_max_length=args.reranker_max_length,
        reranker_trust_remote_code=args.reranker_trust_remote_code,
        oversample_factor=args.oversample_factor,
        candidate_pool_size=args.candidate_pool_size,
        scope_candidates_per_branch=args.scope_candidates_per_branch,
        max_chunks_per_article=args.max_chunks_per_article,
        ambiguous_max_chunks_per_article=args.ambiguous_max_chunks_per_article,
        scope_relevance_margin=args.scope_relevance_margin,
    )
    retriever = (
        build_expanded_retriever(retriever_config, per_query_top_k=args.per_query_top_k)
        if args.retrieval_mode == "auto"
        else build_dense_retriever(retriever_config)
    )
    answer_llm = HuggingFaceCausalLLM(
        HuggingFaceCausalLLMConfig(
            model_name_or_path=args.answer_model,
            device=args.answer_device,
            max_input_tokens=args.answer_max_input_tokens,
            max_new_tokens=args.answer_max_new_tokens,
            trust_remote_code=args.answer_trust_remote_code,
            enable_thinking=False,
            load_in_4bit=args.answer_load_in_4bit,
        )
    )
    use_case = GenerateCompetitionSubmissionUseCase(
        question_source=JsonCompetitionQuestionSource(args.questions),
        retriever=retriever,
        citation_resolver=JsonlLegalCitationResolver(args.articles),
        answer_generator=GroundedLegalAnswerGenerator(
            answer_llm,
            GroundedLegalAnswerGeneratorConfig(
                max_context_characters=args.max_context_characters,
                max_characters_per_context=args.max_characters_per_context,
            ),
        ),
        validator=SubmissionValidator(),
        artifact_writer=SubmissionArtifactWriter(),
        checkpoint=JsonSubmissionCheckpoint(
            args.checkpoint_path or args.output_dir / "partial_results.json"
        ),
    )
    artifact = use_case.execute(
        GenerateSubmissionCommand(
            output_dir=args.output_dir,
            retrieval_top_k=args.retrieval_top_k,
        )
    )
    print(f"Generated {artifact.record_count} records.")
    print(f"results.json: {artifact.results_path}")
    print(f"submission.zip: {artifact.zip_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
