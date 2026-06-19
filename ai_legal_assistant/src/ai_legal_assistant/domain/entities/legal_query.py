from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LegalQuery:
    text: str

    def __post_init__(self) -> None:
        normalized = self.text.strip()
        if not normalized:
            raise ValueError("Legal query cannot be empty.")
        object.__setattr__(self, "text", normalized)
