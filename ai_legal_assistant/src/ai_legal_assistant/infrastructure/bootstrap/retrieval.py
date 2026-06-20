from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ai_legal_assistant.application.ports.embedding_port import QueryEmbeddingPort
from ai_legal_assistant.application.use_cases.retrieve_legal_context import RetrieveLegalContextUseCase
from ai_legal_assistant.application.use_cases.retrieve_expanded_legal_context import (
    RetrieveExpandedLegalContextUseCase,
)
from ai_legal_assistant.domain.services.weighted_rrf import WeightedReciprocalRankFusion
from ai_legal_assistant.infrastructure.bootstrap.query_planning import (
    QueryPlannerConfig,
    build_query_planner,
)
from ai_legal_assistant.infrastructure.embeddings.qwen3_query_embedder import (
    DEFAULT_QUERY_INSTRUCTION,
    Qwen3QueryEmbedder,
    Qwen3QueryEmbedderConfig,
)
from ai_legal_assistant.infrastructure.embeddings.ollama_query_embedder import (
    OllamaQueryEmbedder,
    OllamaQueryEmbedderConfig,
)
from ai_legal_assistant.infrastructure.vectorstores.qdrant_vector_store import QdrantVectorStore
from ai_legal_assistant.infrastructure.search.bm25_search import BM25Search
from ai_legal_assistant.infrastructure.rerankers.cross_encoder_reranker import (
    CrossEncoderReranker,
    CrossEncoderRerankerConfig,
)


@dataclass(frozen=True)
class DenseRetrievalConfig:
    model_name_or_path: str
    embedding_provider: Literal["ollama", "huggingface"] = "huggingface"
    ollama_url: str = "http://localhost:11434"
    collection_name: str = "law_chunks_qwen3_06b"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    vector_size: int = 1024
    query_instruction: str | None = DEFAULT_QUERY_INSTRUCTION
    max_length: int = 768
    device: str | None = None
    trust_remote_code: bool = False
    planner_model_name_or_path: str = "Qwen/Qwen3-0.6B"
    planner_device: str | None = None
    planner_max_input_tokens: int = 4096
    planner_max_new_tokens: int = 500
    planner_trust_remote_code: bool = False
    bm25_index_path: Path | None = None
    bm25_weight: float = 0.7
    reranker_model_name_or_path: str | None = None
    reranker_device: str | None = None
    reranker_batch_size: int = 8
    reranker_max_length: int = 512
    reranker_trust_remote_code: bool = False
    oversample_factor: int = 5
    candidate_pool_size: int = 50
    scope_candidates_per_branch: int = 5
    max_chunks_per_article: int = 2
    ambiguous_max_chunks_per_article: int = 1
    scope_relevance_margin: float = 0.15


def build_dense_retriever(config: DenseRetrievalConfig) -> RetrieveLegalContextUseCase:
    query_embedder, vector_store = _build_retrieval_adapters(config)
    return RetrieveLegalContextUseCase(
        query_embedder=query_embedder,
        vector_store=vector_store,
        expected_vector_size=config.vector_size,
    )


def build_expanded_retriever(
    config: DenseRetrievalConfig,
    *,
    per_query_top_k: int = 20,
) -> RetrieveExpandedLegalContextUseCase:
    query_embedder, vector_store = _build_retrieval_adapters(config)
    sparse_search = (
        BM25Search.from_path(config.bm25_index_path)
        if config.bm25_index_path is not None
        else None
    )
    reranker = (
        CrossEncoderReranker(
            CrossEncoderRerankerConfig(
                model_name_or_path=config.reranker_model_name_or_path,
                device=config.reranker_device,
                batch_size=config.reranker_batch_size,
                max_length=config.reranker_max_length,
                trust_remote_code=config.reranker_trust_remote_code,
            )
        )
        if config.reranker_model_name_or_path is not None
        else None
    )
    return RetrieveExpandedLegalContextUseCase(
        query_planner=build_query_planner(
            QueryPlannerConfig(
                model_name_or_path=config.planner_model_name_or_path,
                device=config.planner_device,
                max_input_tokens=config.planner_max_input_tokens,
                max_new_tokens=config.planner_max_new_tokens,
                trust_remote_code=config.planner_trust_remote_code,
            )
        ),
        query_embedder=query_embedder,
        vector_store=vector_store,
        expected_vector_size=config.vector_size,
        fusion=WeightedReciprocalRankFusion(),
        per_query_top_k=per_query_top_k,
        sparse_search=sparse_search,
        sparse_weight=config.bm25_weight,
        reranker=reranker,
        oversample_factor=config.oversample_factor,
        candidate_pool_size=config.candidate_pool_size,
        scope_candidates_per_branch=config.scope_candidates_per_branch,
        max_chunks_per_article=config.max_chunks_per_article,
        ambiguous_max_chunks_per_article=config.ambiguous_max_chunks_per_article,
        scope_relevance_margin=config.scope_relevance_margin,
    )


def _build_retrieval_adapters(
    config: DenseRetrievalConfig,
) -> tuple[QueryEmbeddingPort, QdrantVectorStore]:
    if config.embedding_provider == "ollama":
        query_embedder = OllamaQueryEmbedder(
            OllamaQueryEmbedderConfig(
                model=config.model_name_or_path,
                base_url=config.ollama_url,
                query_instruction=config.query_instruction,
            )
        )
    elif config.embedding_provider == "huggingface":
        query_embedder = Qwen3QueryEmbedder(
            Qwen3QueryEmbedderConfig(
                model_name_or_path=config.model_name_or_path,
                query_instruction=config.query_instruction,
                max_length=config.max_length,
                device=config.device,
                trust_remote_code=config.trust_remote_code,
            )
        )
    else:
        raise ValueError(f"Unsupported embedding provider: {config.embedding_provider}")
    vector_store = QdrantVectorStore(
        collection_name=config.collection_name,
        url=config.qdrant_url,
        api_key=config.qdrant_api_key,
    )
    return query_embedder, vector_store
