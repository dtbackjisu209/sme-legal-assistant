from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ai_legal_assistant.application.dto.retrieval_dto import DenseSearchResult
from ai_legal_assistant.application.dto.vector_store_dto import VectorPoint


class QdrantVectorStore:
    KEYWORD_INDEX_FIELDS = (
        "chunk_id",
        "article_id",
        "subject_id",
        "topic_id",
        "clause_number",
        "point_label",
        "chunk_type",
        "parent_chunk_id",
    )
    INTEGER_INDEX_FIELDS = (
        "subject_number",
        "topic_number",
        "ordinal",
        "subchunk_index",
        "subchunk_count",
    )
    BOOL_INDEX_FIELDS = ("is_subchunk",)

    def __init__(
        self,
        collection_name: str,
        *,
        url: str = "http://localhost:6333",
        api_key: str | None = None,
        wait: bool = True,
        timeout_seconds: float = 120.0,
    ) -> None:
        self.collection_name = collection_name
        self.wait = wait
        self._url = url
        self._api_key = api_key
        self._timeout_seconds = timeout_seconds
        self._client: Any | None = None
        self._models: Any | None = None

    def ensure_collection(self, vector_size: int, recreate: bool = False) -> None:
        if vector_size <= 0:
            raise ValueError("vector_size must be positive.")

        client, models = self._client_and_models()
        exists = client.collection_exists(self.collection_name)
        if exists and recreate:
            client.delete_collection(collection_name=self.collection_name)
            exists = False

        if not exists:
            client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE,
                ),
            )
            return

        existing_size = self._get_existing_vector_size()
        if existing_size is not None and existing_size != vector_size:
            raise ValueError(
                f"Qdrant collection '{self.collection_name}' has vector size {existing_size}, "
                f"but embedding model returns {vector_size}. Use --recreate or a different collection."
            )

    def create_payload_indexes(self) -> None:
        _, models = self._client_and_models()
        for field_name in self.KEYWORD_INDEX_FIELDS:
            self._create_payload_index(field_name, models.PayloadSchemaType.KEYWORD)
        for field_name in self.INTEGER_INDEX_FIELDS:
            self._create_payload_index(field_name, models.PayloadSchemaType.INTEGER)
        bool_schema = getattr(models.PayloadSchemaType, "BOOL", None)
        if bool_schema is None:
            return
        for field_name in self.BOOL_INDEX_FIELDS:
            self._create_payload_index(field_name, bool_schema)

    def upsert(self, points: Sequence[VectorPoint]) -> None:
        if not points:
            return

        client, models = self._client_and_models()
        client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=point.point_id,
                    vector=point.vector,
                    payload=point.payload,
                )
                for point in points
            ],
            wait=self.wait,
        )

    def search(self, query_vector: Sequence[float], top_k: int = 10) -> list[DenseSearchResult]:
        if len(query_vector) == 0:
            raise ValueError("query_vector cannot be empty.")
        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        client, _ = self._client_and_models()
        response = client.query_points(
            collection_name=self.collection_name,
            query=list(query_vector),
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )
        results: list[DenseSearchResult] = []
        for point in response.points:
            payload = dict(point.payload or {})
            text = str(payload.pop("text", ""))
            chunk_id = str(payload.get("chunk_id") or point.id)
            results.append(
                DenseSearchResult(
                    chunk_id=chunk_id,
                    score=float(point.score),
                    text=text,
                    metadata=payload,
                )
            )
        return results

    def set_indexing_threshold(self, threshold: int) -> None:
        if threshold < 0:
            raise ValueError("indexing threshold cannot be negative.")
        client, models = self._client_and_models()
        client.update_collection(
            collection_name=self.collection_name,
            optimizer_config=models.OptimizersConfigDiff(indexing_threshold=threshold),
        )

    def count(self) -> int:
        client, _ = self._client_and_models()
        return int(
            client.count(
                collection_name=self.collection_name,
                exact=True,
            ).count
        )

    def _create_payload_index(self, field_name: str, schema: object) -> None:
        client, _ = self._client_and_models()
        try:
            client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field_name,
                field_schema=schema,
                wait=self.wait,
            )
        except Exception as exc:
            message = str(exc).lower()
            if "already exists" in message or "already created" in message:
                return
            raise

    def _get_existing_vector_size(self) -> int | None:
        client, _ = self._client_and_models()
        collection = client.get_collection(collection_name=self.collection_name)
        vectors_config = collection.config.params.vectors
        if isinstance(vectors_config, dict):
            first_vector = next(iter(vectors_config.values()), None)
            return getattr(first_vector, "size", None)
        return getattr(vectors_config, "size", None)

    def _client_and_models(self) -> tuple[Any, Any]:
        if self._client is None or self._models is None:
            try:
                from qdrant_client import QdrantClient
                from qdrant_client.http import models
            except ImportError as exc:
                raise RuntimeError("Missing dependency: qdrant-client.") from exc

            self._models = models
            self._client = QdrantClient(
                url=self._url,
                api_key=self._api_key,
                timeout=self._timeout_seconds,
            )
        return self._client, self._models
