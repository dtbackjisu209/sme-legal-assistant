from __future__ import annotations

from dataclasses import dataclass

from ai_legal_assistant.application.dto.retrieval_dto import (
    DenseSearchResult,
    RetrieveLegalContextQuery,
)
from ai_legal_assistant.application.ports.embedding_port import QueryEmbeddingPort
from ai_legal_assistant.application.ports.vector_store_port import VectorSearchPort
from ai_legal_assistant.domain.entities.legal_query import LegalQuery


@dataclass
class RetrieveLegalContextUseCase:
    query_embedder: QueryEmbeddingPort
    vector_store: VectorSearchPort
    expected_vector_size: int

    def execute(self, request: RetrieveLegalContextQuery) -> list[DenseSearchResult]:
        if request.top_k <= 0:
            raise ValueError("top_k must be positive.")

        vector = list(self.query_embedder.embed_query(LegalQuery(request.query)))
        if len(vector) != self.expected_vector_size:
            raise ValueError(
                f"Query embedding has dimension {len(vector)}; "
                f"Qdrant collection expects {self.expected_vector_size}."
            )
        return self.vector_store.search(vector, top_k=request.top_k)
