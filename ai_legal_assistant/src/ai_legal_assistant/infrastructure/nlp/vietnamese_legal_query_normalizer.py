from __future__ import annotations

import re
import unicodedata

from ai_legal_assistant.domain.entities.legal_query import LegalQuery


class VietnameseLegalQueryNormalizer:
    ABBREVIATIONS = (
        (re.compile(r"\bTNHH\b", re.IGNORECASE), "trách nhiệm hữu hạn"),
        (re.compile(r"\bBHXH\b", re.IGNORECASE), "bảo hiểm xã hội"),
        (re.compile(r"\bGTGT\b", re.IGNORECASE), "giá trị gia tăng"),
    )
    LEGAL_REFERENCE_PATTERNS = (
        (re.compile(r"\bđiều\s*([0-9]+(?:\.[0-9A-Za-zĐđ]+)*)", re.IGNORECASE), "Điều"),
        (re.compile(r"\bkhoản\s*([0-9]+)", re.IGNORECASE), "Khoản"),
        (re.compile(r"\bđiểm\s*([A-Za-zĐđ])", re.IGNORECASE), "Điểm"),
    )
    DOCUMENT_NUMBER_PATTERN = re.compile(
        r"\b(\d+)\s*/\s*(\d{4})\s*/\s*([A-Za-zĐđ]+)\s*-\s*([A-Za-zĐđ]+)\b",
        re.IGNORECASE,
    )

    def normalize(self, query: LegalQuery) -> LegalQuery:
        value = unicodedata.normalize("NFC", query.text)
        value = re.sub(r"\s+", " ", value).strip()
        for pattern, replacement in self.ABBREVIATIONS:
            value = pattern.sub(replacement, value)
        for pattern, label in self.LEGAL_REFERENCE_PATTERNS:
            value = pattern.sub(lambda match: f"{label} {match.group(1)}", value)
        value = self.DOCUMENT_NUMBER_PATTERN.sub(self._normalize_document_number, value)
        return LegalQuery(value)

    @staticmethod
    def _normalize_document_number(match: re.Match[str]) -> str:
        suffix = f"{match.group(3)}-{match.group(4)}".upper()
        return f"{match.group(1)}/{match.group(2)}/{suffix}"
