from __future__ import annotations

from dataclasses import dataclass
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


def build_dense_retriever(config: DenseRetrievalConfig) -> RetrieveLegalContextUseCase:
    query_embedder, vector_store = _build_retrieval_adapters(config)
    return RetrieveLegalContextUseCase(
        query_embedder=query_embedder,
        vector_store=vector_store,
        expected_vector_size=config.vector_size,
    )


def build_expanded_dense_retriever(
    config: DenseRetrievalConfig,
    *,
    per_query_top_k: int = 20,
) -> RetrieveExpandedLegalContextUseCase:
    query_embedder, vector_store = _build_retrieval_adapters(config)
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
