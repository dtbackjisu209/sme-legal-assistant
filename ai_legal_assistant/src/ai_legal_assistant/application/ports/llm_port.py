from __future__ import annotations

from typing import Any, Protocol


class TextGenerationPort(Protocol):
    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        forbidden_phrases: tuple[str, ...] = (),
        json_schema: dict[str, Any] | None = None,
    ) -> str:
        ...
