from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

from ai_legal_assistant.domain.entities.bm25_index import BM25Index, SearchableLawChunk
from ai_legal_assistant.domain.services.text_tokenizer import TextTokenizer


@dataclass
class BM25IndexBuilder:
    tokenizer: TextTokenizer
    k1: float = 1.5
    b: float = 0.75

    def __post_init__(self) -> None:
        if self.k1 <= 0:
            raise ValueError("k1 must be positive.")
        if not 0 <= self.b <= 1:
            raise ValueError("b must be between 0 and 1.")

    def build(self, chunks: Iterable[SearchableLawChunk]) -> BM25Index:
        postings: dict[str, list[tuple[int, int]]] = {}
        doc_lengths: list[int] = []
        chunk_ids: list[str] = []
        texts: list[str] = []
        metadatas: list[dict[str, object]] = []

        for doc_index, chunk in enumerate(chunks):
            tokens = self.tokenizer.tokenize(chunk.text)
            term_frequency = Counter(tokens)
            doc_lengths.append(sum(term_frequency.values()))
            chunk_ids.append(chunk.chunk_id)
            texts.append(chunk.text)
            metadatas.append(dict(chunk.metadata))

            for token, frequency in term_frequency.items():
                postings.setdefault(token, []).append((doc_index, frequency))

        document_count = len(chunk_ids)
        avgdl = sum(doc_lengths) / document_count if document_count else 0.0
        idf = {
            token: self._idf(document_count=document_count, document_frequency=len(token_postings))
            for token, token_postings in postings.items()
        }

        return BM25Index(
            postings=postings,
            idf=idf,
            doc_lengths=doc_lengths,
            avgdl=avgdl,
            chunk_ids=chunk_ids,
            texts=texts,
            metadatas=metadatas,
            k1=self.k1,
            b=self.b,
        )

    def _idf(self, document_count: int, document_frequency: int) -> float:
        if document_count == 0 or document_frequency == 0:
            return 0.0
        return math.log(1 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5))
