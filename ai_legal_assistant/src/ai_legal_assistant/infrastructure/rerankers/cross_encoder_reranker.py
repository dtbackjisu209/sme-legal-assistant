from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ai_legal_assistant.domain.entities.retrieval_candidate import (
    RerankedCandidate,
    RetrievalCandidate,
)


@dataclass(frozen=True)
class CrossEncoderRerankerConfig:
    model_name_or_path: str
    device: str | None = None
    batch_size: int = 8
    max_length: int = 512
    trust_remote_code: bool = False


class CrossEncoderReranker:
    def __init__(self, config: CrossEncoderRerankerConfig) -> None:
        if config.batch_size <= 0:
            raise ValueError("Reranker batch_size must be positive.")
        if config.max_length <= 0:
            raise ValueError("Reranker max_length must be positive.")
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError("Missing dependency: sentence-transformers.") from exc

        self.config = config
        self._model = CrossEncoder(
            config.model_name_or_path,
            device=config.device,
            max_length=config.max_length,
            trust_remote_code=config.trust_remote_code,
        )

    def rerank(
        self,
        *,
        query: str,
        intent: str,
        scope_queries: dict[str, str],
        candidates: tuple[RetrievalCandidate, ...],
    ) -> tuple[RerankedCandidate, ...]:
        if not candidates:
            return ()
        pairs = [
            (
                self._query_context(query, intent, candidate, scope_queries),
                self._document_context(candidate),
            )
            for candidate in candidates
        ]
        raw_scores = self._model.predict(
            pairs,
            batch_size=self.config.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        values: Any = raw_scores.tolist() if hasattr(raw_scores, "tolist") else raw_scores
        return tuple(
            RerankedCandidate(candidate=candidate, relevance_score=self._score(value))
            for candidate, value in zip(candidates, values, strict=True)
        )

    @staticmethod
    def _query_context(
        query: str,
        intent: str,
        candidate: RetrievalCandidate,
        scope_queries: dict[str, str],
    ) -> str:
        scope_context = "\n".join(
            scope_queries[scope]
            for scope in candidate.matched_scopes
            if scope in scope_queries
        )
        return (
            f"Câu hỏi pháp luật: {query}\n"
            f"Ý định cần trả lời: {intent}\n"
            f"Câu truy vấn theo phạm vi: {scope_context}"
        )

    @staticmethod
    def _document_context(candidate: RetrievalCandidate) -> str:
        article_title = str(candidate.metadata.get("article_title") or "")
        scopes = ", ".join(candidate.matched_scopes)
        return (
            f"Điều luật: {article_title}\n"
            f"Phạm vi truy xuất: {scopes}\n"
            f"Nội dung: {candidate.text}"
        )

    @staticmethod
    def _score(value: Any) -> float:
        if isinstance(value, (list, tuple)):
            if not value:
                raise ValueError("Reranker returned an empty score vector.")
            value = value[-1]
        return float(value)
