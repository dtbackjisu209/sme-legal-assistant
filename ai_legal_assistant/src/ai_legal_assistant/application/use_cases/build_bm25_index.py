from __future__ import annotations

from dataclasses import dataclass

from ai_legal_assistant.application.dto.retrieval_dto import BuildBM25IndexCommand, BuildBM25IndexResult
from ai_legal_assistant.application.ports.bm25_index_port import BM25IndexStorePort, LawChunkReaderPort
from ai_legal_assistant.domain.services.bm25_index_builder import BM25IndexBuilder


@dataclass
class BuildBM25IndexUseCase:
    chunk_reader: LawChunkReaderPort
    index_store: BM25IndexStorePort
    index_builder: BM25IndexBuilder

    def execute(self, command: BuildBM25IndexCommand) -> BuildBM25IndexResult:
        index = self.index_builder.build(self.chunk_reader.iter_chunks(limit=command.limit))
        self.index_store.save(index)
        return BuildBM25IndexResult(
            document_count=index.document_count,
            vocabulary_size=index.vocabulary_size,
            avgdl=index.avgdl,
            output_path=command.output_path,
        )
