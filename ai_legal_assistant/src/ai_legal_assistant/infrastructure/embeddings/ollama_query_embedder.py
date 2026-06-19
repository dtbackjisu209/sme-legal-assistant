from __future__ import annotations

import json
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ai_legal_assistant.domain.entities.legal_query import LegalQuery
from ai_legal_assistant.infrastructure.embeddings.qwen3_query_embedder import (
    DEFAULT_QUERY_INSTRUCTION,
)


@dataclass(frozen=True)
class OllamaQueryEmbedderConfig:
    model: str = "qwen3-embedding:0.6b"
    base_url: str = "http://localhost:11434"
    query_instruction: str | None = DEFAULT_QUERY_INSTRUCTION
    timeout_seconds: float = 120.0


class OllamaQueryEmbedder:
    def __init__(self, config: OllamaQueryEmbedderConfig) -> None:
        if not config.model.strip():
            raise ValueError("Ollama model cannot be empty.")
        if config.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")
        self.config = config

    def embed_query(self, query: LegalQuery) -> list[float]:
        body = json.dumps(
            {
                "model": self.config.model,
                "input": self._format_query(query.text),
                "truncate": True,
            }
        ).encode("utf-8")
        request = Request(
            f"{self.config.base_url.rstrip('/')}/api/embed",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.config.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama embedding request failed ({exc.code}): {detail}") from exc
        except URLError as exc:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.config.base_url}. "
                "Start the Ollama service before querying."
            ) from exc

        embeddings = payload.get("embeddings")
        if not isinstance(embeddings, list) or not embeddings or not isinstance(embeddings[0], list):
            raise RuntimeError("Ollama returned no embedding vector.")
        return [float(value) for value in embeddings[0]]

    def _format_query(self, query: str) -> str:
        instruction = self.config.query_instruction
        if instruction is None or not instruction.strip():
            return query
        return f"Instruct: {instruction.strip()}\nQuery:{query}"
