from __future__ import annotations

from pathlib import Path

from ai_legal_assistant.application.dto.retrieval_dto import SparseSearchResult
from ai_legal_assistant.domain.entities.bm25_index import BM25Index
from ai_legal_assistant.domain.services.bm25_scorer import BM25Scorer
from ai_legal_assistant.domain.services.text_tokenizer import TextTokenizer
from ai_legal_assistant.infrastructure.nlp.vietnamese_legal_tokenizer import VietnameseLegalTokenizer
from ai_legal_assistant.infrastructure.search.bm25_index_store import PickleBM25IndexStore


class BM25Search:
    def __init__(self, index: BM25Index, tokenizer: TextTokenizer) -> None:
        self.index = index
        self.tokenizer = tokenizer
        self.scorer = BM25Scorer(index)

    @classmethod
    def from_path(cls, index_path: Path) -> "BM25Search":
        payload = PickleBM25IndexStore(index_path).load_payload()
        tokenizer = VietnameseLegalTokenizer.from_config(payload.tokenizer_config)
        return cls(index=payload.index, tokenizer=tokenizer)

    def search(self, query: str, top_k: int = 10) -> list[SparseSearchResult]:
        return [
            SparseSearchResult(
                chunk_id=self.index.chunk_ids[scored.doc_index],
                score=scored.score,
                text=self.index.texts[scored.doc_index],
                metadata=self.index.metadatas[scored.doc_index],
            )
            for scored in self.scorer.score(self.tokenizer.tokenize(query), top_k=top_k)
        ]
