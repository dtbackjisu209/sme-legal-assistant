from __future__ import annotations

from collections.abc import Sequence

from ai_legal_assistant.application.ports.vector_store_port import VectorPoint


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
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http import models
        except ImportError as exc:
            raise RuntimeError("Missing dependency: qdrant-client.") from exc

        self.collection_name = collection_name
        self.wait = wait
        self._models = models
        self._client = QdrantClient(url=url, api_key=api_key, timeout=timeout_seconds)

    def ensure_collection(self, vector_size: int, recreate: bool = False) -> None:
        if vector_size <= 0:
            raise ValueError("vector_size must be positive.")

        exists = self._client.collection_exists(self.collection_name)
        if exists and recreate:
            self._client.delete_collection(collection_name=self.collection_name)
            exists = False

        if not exists:
            self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=self._models.VectorParams(
                    size=vector_size,
                    distance=self._models.Distance.COSINE,
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
        for field_name in self.KEYWORD_INDEX_FIELDS:
            self._create_payload_index(field_name, self._models.PayloadSchemaType.KEYWORD)
        for field_name in self.INTEGER_INDEX_FIELDS:
            self._create_payload_index(field_name, self._models.PayloadSchemaType.INTEGER)
        bool_schema = getattr(self._models.PayloadSchemaType, "BOOL", None)
        if bool_schema is None:
            return
        for field_name in self.BOOL_INDEX_FIELDS:
            self._create_payload_index(field_name, bool_schema)

    def upsert(self, points: Sequence[VectorPoint]) -> None:
        if not points:
            return

        self._client.upsert(
            collection_name=self.collection_name,
            points=[
                self._models.PointStruct(
                    id=point.point_id,
                    vector=point.vector,
                    payload=point.payload,
                )
                for point in points
            ],
            wait=self.wait,
        )

    def set_indexing_threshold(self, threshold: int) -> None:
        if threshold < 0:
            raise ValueError("indexing threshold cannot be negative.")
        self._client.update_collection(
            collection_name=self.collection_name,
            optimizer_config=self._models.OptimizersConfigDiff(indexing_threshold=threshold),
        )

    def count(self) -> int:
        return int(
            self._client.count(
                collection_name=self.collection_name,
                exact=True,
            ).count
        )

    def _create_payload_index(self, field_name: str, schema: object) -> None:
        try:
            self._client.create_payload_index(
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
        collection = self._client.get_collection(collection_name=self.collection_name)
        vectors_config = collection.config.params.vectors
        if isinstance(vectors_config, dict):
            first_vector = next(iter(vectors_config.values()), None)
            return getattr(first_vector, "size", None)
        return getattr(vectors_config, "size", None)
