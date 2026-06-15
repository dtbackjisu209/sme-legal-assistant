from __future__ import annotations

import heapq
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

from ai_legal_assistant.domain.entities.bm25_index import BM25Index


@dataclass(frozen=True)
class BM25ScoredDocument:
    doc_index: int
    score: float


@dataclass(frozen=True)
class BM25Scorer:
    index: BM25Index

    def score(self, query_tokens: Iterable[str], top_k: int = 10) -> list[BM25ScoredDocument]:
        if top_k <= 0:
            return []
        if self.index.document_count == 0 or self.index.avgdl == 0:
            return []

        scores: dict[int, float] = {}
        for token, query_frequency in Counter(query_tokens).items():
            idf = self.index.idf.get(token)
            postings = self.index.postings.get(token)
            if idf is None or not postings:
                continue

            for doc_index, term_frequency in postings:
                scores[doc_index] = scores.get(doc_index, 0.0) + query_frequency * self._token_score(
                    doc_index=doc_index,
                    term_frequency=term_frequency,
                    idf=idf,
                )

        ranked = heapq.nlargest(top_k, scores.items(), key=lambda item: item[1])
        return [BM25ScoredDocument(doc_index=doc_index, score=score) for doc_index, score in ranked]

    def _token_score(self, doc_index: int, term_frequency: int, idf: float) -> float:
        doc_length = self.index.doc_lengths[doc_index]
        denominator = term_frequency + self.index.k1 * (
            1 - self.index.b + self.index.b * doc_length / self.index.avgdl
        )
        return idf * (term_frequency * (self.index.k1 + 1)) / denominator
