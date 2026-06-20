from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalContentIdentity:
    version: str = "v2"

    def create(self, *, article_id: str | None, text: str) -> str:
        normalized_text = self.normalize_text(text)
        if not normalized_text:
            raise ValueError("Cannot create content identity for empty text.")
        canonical = f"{self.version}|{(article_id or '').strip()}|{normalized_text}"
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]

    @staticmethod
    def normalize_text(text: str) -> str:
        value = unicodedata.normalize("NFC", str(text or ""))
        return re.sub(r"\s+", " ", value).strip()

