from __future__ import annotations

from collections.abc import Sequence
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

    def generate_batch(
        self,
        *,
        system_prompt: str,
        user_prompts: Sequence[str],
        forbidden_phrases: tuple[str, ...] = (),
    ) -> tuple[str, ...]:
        ...
