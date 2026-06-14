from __future__ import annotations

from ai_legal_assistant.application.dto.ingestion_dto import IngestPhapdienCommand, IngestPhapdienResult
from ai_legal_assistant.application.ports.dataset_loader_port import DatasetLoaderPort
from ai_legal_assistant.application.ports.law_repository_port import LawRepositoryPort
from ai_legal_assistant.domain.services.legal_chunking_policy import LegalChunkingPolicy


class IngestPhapdienUseCase:
    def __init__(
        self,
        dataset_loader: DatasetLoaderPort,
        chunking_policy: LegalChunkingPolicy,
        law_repository: LawRepositoryPort,
        command: IngestPhapdienCommand | None = None,
    ) -> None:
        self.dataset_loader = dataset_loader
        self.chunking_policy = chunking_policy
        self.law_repository = law_repository
        self.command = command or IngestPhapdienCommand()

    def execute(self, command: IngestPhapdienCommand | None = None) -> IngestPhapdienResult:
        effective_command = command or self.command
        article_count = 0
        chunk_count = 0

        self.law_repository.open()
        try:
            for article in self.dataset_loader.load_articles(limit=effective_command.limit):
                self.law_repository.save_article(article)
                article_count += 1

                for chunk in self.chunking_policy.chunk(article):
                    self.law_repository.save_chunk(chunk)
                    chunk_count += 1
        finally:
            self.law_repository.close()

        return IngestPhapdienResult(
            article_count=article_count,
            chunk_count=chunk_count,
            articles_path=self.law_repository.articles_path,
            chunks_path=self.law_repository.chunks_path,
        )
