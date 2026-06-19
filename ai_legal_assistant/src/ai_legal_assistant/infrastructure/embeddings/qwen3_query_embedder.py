from __future__ import annotations

from dataclasses import dataclass
from ai_legal_assistant.domain.entities.legal_query import LegalQuery


DEFAULT_QUERY_INSTRUCTION = (
    "Given a Vietnamese legal question, retrieve relevant Vietnamese legal passages "
    "that answer the question"
)


@dataclass(frozen=True)
class Qwen3QueryEmbedderConfig:
    model_name_or_path: str
    query_instruction: str | None = DEFAULT_QUERY_INSTRUCTION
    max_length: int = 768
    device: str | None = None
    trust_remote_code: bool = False


class Qwen3QueryEmbedder:
    """SentenceTransformers adapter matching the corpus embedding pipeline."""

    def __init__(self, config: Qwen3QueryEmbedderConfig) -> None:
        if not config.model_name_or_path.strip():
            raise ValueError("model_name_or_path cannot be empty.")
        if config.max_length <= 0:
            raise ValueError("max_length must be positive.")

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Qwen3 query embedding requires sentence-transformers."
            ) from exc

        self.config = config
        self._model = SentenceTransformer(
            config.model_name_or_path,
            device=config.device,
            trust_remote_code=config.trust_remote_code,
        )
        self._model.max_seq_length = config.max_length

    def embed_query(self, query: LegalQuery) -> list[float]:
        text = self._format_query(query.text)
        embeddings = self._model.encode(
            [text],
            batch_size=1,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embeddings[0].astype("float32").tolist()

    def _format_query(self, query: str) -> str:
        instruction = self.config.query_instruction
        if instruction is None or not instruction.strip():
            return query
        return f"Instruct: {instruction.strip()}\nQuery:{query}"
