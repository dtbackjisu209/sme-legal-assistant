from __future__ import annotations

"""Vietnamese legal tokenizer backed by optional infrastructure libraries."""

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Callable, Iterable


DEFAULT_LEGAL_PHRASES: tuple[str, ...] = (
    "bảo hiểm xã hội",
    "bảo hiểm y tế",
    "bảo hiểm thất nghiệp",
    "bồi thường thiệt hại",
    "chấm dứt hợp đồng",
    "chủ sở hữu",
    "chuyển nhượng cổ phần",
    "chuyển nhượng vốn",
    "cơ quan nhà nước",
    "công bố thông tin",
    "công chức",
    "công ty cổ phần",
    "công ty hợp danh",
    "công ty mẹ",
    "công ty trách nhiệm hữu hạn",
    "công ty",
    "đăng ký doanh nghiệp",
    "đại diện theo pháp luật",
    "điều lệ công ty",
    "điều lệ",
    "đình chỉ hoạt động",
    "doanh nghiệp nhỏ và vừa",
    "doanh nghiệp tư nhân",
    "doanh nghiệp",
    "giao dịch dân sự",
    "giấy chứng nhận đăng ký doanh nghiệp",
    "giấy chứng nhận quyền sử dụng đất",
    "góp vốn",
    "hóa đơn điện tử",
    "hóa đơn",
    "hộ kinh doanh",
    "hợp đồng lao động",
    "hợp đồng thương mại",
    "hợp đồng",
    "kê khai thuế",
    "kiểm tra thuế",
    "lao động",
    "mã số doanh nghiệp",
    "mã số thuế",
    "người đại diện theo pháp luật",
    "người lao động",
    "nghĩa vụ thuế",
    "pháp nhân thương mại",
    "phạt vi phạm",
    "quyền sở hữu",
    "quyền sử dụng đất",
    "sở hữu trí tuệ",
    "tạm ngừng kinh doanh",
    "tiền lương",
    "tranh chấp lao động",
    "trách nhiệm hữu hạn",
    "vốn điều lệ",
    "vốn góp",
    "xử phạt hành chính",
)


TokenizeFn = Callable[[str], list[str]]


@dataclass(frozen=True)
class VietnameseLegalTokenizerConfig:
    backend: str = "auto"
    legal_phrases: tuple[str, ...] = DEFAULT_LEGAL_PHRASES
    phrase_boost: int = 1
    lowercase: bool = True
    repair_mojibake: bool = True

    def to_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "legal_phrases": list(self.legal_phrases),
            "phrase_boost": self.phrase_boost,
            "lowercase": self.lowercase,
            "repair_mojibake": self.repair_mojibake,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> "VietnameseLegalTokenizerConfig":
        phrases = raw.get("legal_phrases", DEFAULT_LEGAL_PHRASES)
        return cls(
            backend=str(raw.get("backend", "auto")),
            legal_phrases=tuple(str(item) for item in phrases) if isinstance(phrases, list) else DEFAULT_LEGAL_PHRASES,
            phrase_boost=int(raw.get("phrase_boost", 1)),
            lowercase=bool(raw.get("lowercase", True)),
            repair_mojibake=bool(raw.get("repair_mojibake", True)),
        )


@dataclass
class VietnameseLegalTokenizer:
    config: VietnameseLegalTokenizerConfig = field(default_factory=VietnameseLegalTokenizerConfig)

    _TOKEN_RE = re.compile(r"[0-9A-Za-zÀ-ỹ_]+", re.UNICODE)
    _ENCODED_RUN_RE = re.compile(r"[\u0000-\u00ff\u2018\u2019\u201c\u201d\u2026\u2039\u203a]+")
    _MOJIBAKE_MARKERS = (
        "\u00c3",
        "\u00c2",
        "\u00c4",
        "\u00c6",
        "\u00ca",
        "\u00e1\u00ba",
        "\u00e1\u00bb",
        "\u00c4\u0091",
        "\u00c4\u2018",
        "\u00c4\u0090",
    )

    def __post_init__(self) -> None:
        if self.config.phrase_boost < 0:
            raise ValueError("phrase_boost cannot be negative.")

        tokenize, backend_name = self._resolve_backend(self.config.backend)
        self._tokenize_backend: TokenizeFn = tokenize
        self.backend_name = backend_name
        self._phrase_patterns = tuple(
            (self._phrase_token(phrase), self._compile_phrase_pattern(phrase))
            for phrase in self.config.legal_phrases
            if phrase.strip()
        )

    @classmethod
    def from_config(cls, raw: dict[str, object]) -> "VietnameseLegalTokenizer":
        return cls(VietnameseLegalTokenizerConfig.from_dict(raw))

    def tokenize(self, text: str) -> list[str]:
        normalized = self.normalize(text)
        if not normalized:
            return []

        tokens = self._tokenize_backend(normalized)
        phrase_tokens = self._match_legal_phrases(normalized)
        if self.config.phrase_boost:
            tokens.extend(phrase_tokens * self.config.phrase_boost)
        return [token for token in tokens if token]

    def normalize(self, text: str) -> str:
        value = "" if text is None else str(text)
        if self.config.repair_mojibake:
            value = self._repair_mojibake(value)

        value = unicodedata.normalize("NFC", value)
        value = value.replace("\ufeff", " ").replace("\u00a0", " ")
        value = value.replace("\r\n", "\n").replace("\r", "\n")
        value = re.sub(r"[ \t]+", " ", value)
        value = re.sub(r"\n+", "\n", value)
        value = value.strip()
        if self.config.lowercase:
            value = value.lower()
        return value

    def export_config(self) -> dict[str, object]:
        config = self.config.to_dict()
        config["backend"] = self.backend_name
        return config

    def _resolve_backend(self, backend: str) -> tuple[TokenizeFn, str]:
        if backend not in {"auto", "underthesea", "pyvi", "regex"}:
            raise ValueError("backend must be one of: auto, underthesea, pyvi, regex.")

        if backend in {"auto", "underthesea"}:
            underthesea = self._try_underthesea()
            if underthesea is not None:
                return underthesea, "underthesea"
            if backend == "underthesea":
                raise RuntimeError("Missing dependency: underthesea.")

        if backend in {"auto", "pyvi"}:
            pyvi = self._try_pyvi()
            if pyvi is not None:
                return pyvi, "pyvi"
            if backend == "pyvi":
                raise RuntimeError("Missing dependency: pyvi.")

        return self._regex_tokenize, "regex"

    def _try_underthesea(self) -> TokenizeFn | None:
        try:
            from underthesea import word_tokenize
        except ImportError:
            return None

        def tokenize(text: str) -> list[str]:
            try:
                raw = word_tokenize(text, format="text")
            except TypeError:
                raw = word_tokenize(text)
            return self._clean_backend_tokens(raw.split() if isinstance(raw, str) else raw)

        return tokenize

    def _try_pyvi(self) -> TokenizeFn | None:
        try:
            from pyvi import ViTokenizer
        except ImportError:
            return None

        def tokenize(text: str) -> list[str]:
            return self._clean_backend_tokens(ViTokenizer.tokenize(text).split())

        return tokenize

    def _regex_tokenize(self, text: str) -> list[str]:
        return self._clean_backend_tokens(self._TOKEN_RE.findall(text))

    def _clean_backend_tokens(self, tokens: Iterable[object]) -> list[str]:
        cleaned: list[str] = []
        for token in tokens:
            value = str(token).strip(".,;:!?()[]{}\"'")
            value = value.replace(" ", "_")
            if value and self._TOKEN_RE.fullmatch(value):
                cleaned.append(value)
        return cleaned

    def _match_legal_phrases(self, text: str) -> list[str]:
        matches: list[str] = []
        for token, pattern in self._phrase_patterns:
            if pattern.search(text):
                matches.append(token)
        return matches

    def _compile_phrase_pattern(self, phrase: str) -> re.Pattern[str]:
        normalized = self.normalize(phrase)
        parts = [re.escape(part) for part in normalized.split()]
        return re.compile(r"(?<!\w)" + r"\s+".join(parts) + r"(?!\w)", re.UNICODE)

    def _phrase_token(self, phrase: str) -> str:
        return "_".join(self.normalize(phrase).split())

    def _repair_mojibake(self, text: str) -> str:
        if not self._looks_mojibake(text):
            return text

        repaired = self._repair_encoded_run(text)
        if repaired != text:
            return repaired

        return self._ENCODED_RUN_RE.sub(lambda match: self._repair_encoded_run(match.group(0)), text)

    def _repair_encoded_run(self, text: str) -> str:
        best_text = text
        best_quality = self._text_quality(text)

        for encoding in ("latin1", "cp1252"):
            try:
                repaired = text.encode(encoding).decode("utf-8")
            except UnicodeError:
                continue
            quality = self._text_quality(repaired)
            if quality > best_quality:
                best_text = repaired
                best_quality = quality

        return best_text

    def _looks_mojibake(self, text: str) -> bool:
        return any(marker in text for marker in self._MOJIBAKE_MARKERS) or any(
            "\u0080" <= char <= "\u009f" for char in text
        )

    def _text_quality(self, text: str) -> int:
        return self._vietnamese_score(text) - 10 * self._mojibake_badness(text)

    def _vietnamese_score(self, text: str) -> int:
        return sum(1 for char in text if "À" <= char <= "ỹ")

    def _mojibake_badness(self, text: str) -> int:
        marker_count = sum(text.count(marker) for marker in self._MOJIBAKE_MARKERS)
        control_count = sum(1 for char in text if "\u0080" <= char <= "\u009f")
        return marker_count + control_count
