from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ai_legal_assistant.application.dto.ingestion_dto import IngestPhapdienCommand
from ai_legal_assistant.application.dto.vector_import_dto import ImportParquetVectorStoreCommand
from ai_legal_assistant.application.use_cases.ingest_phapdien import IngestPhapdienUseCase
from ai_legal_assistant.application.use_cases.import_parquet_vector_store import (
    ImportParquetVectorStoreUseCase,
)
from ai_legal_assistant.domain.services.legal_chunking_policy import LegalChunkingPolicy
from ai_legal_assistant.infrastructure.datasets.huggingface_phapdien_loader import HuggingFacePhapdienLoader
from ai_legal_assistant.infrastructure.persistence.jsonl_law_repository import JsonlLawRepository
from ai_legal_assistant.infrastructure.persistence.parquet_vector_shard_reader import (
    ParquetVectorShardReader,
)
from ai_legal_assistant.infrastructure.progress.console_progress_reporter import ConsoleProgressReporter
from ai_legal_assistant.infrastructure.vectorstores.qdrant_vector_store import QdrantVectorStore


@dataclass(frozen=True)
class Container:
    ingest_phapdien_use_case: IngestPhapdienUseCase


@dataclass(frozen=True)
class ImportVectorStoreContainer:
    import_vector_store_use_case: ImportParquetVectorStoreUseCase


def build_container(output_dir: Path = Path("data/processed"), limit: int | None = None) -> Container:
    command = IngestPhapdienCommand(output_dir=output_dir, limit=limit)
    dataset_loader = HuggingFacePhapdienLoader()
    chunking_policy = LegalChunkingPolicy()
    law_repository = JsonlLawRepository(output_dir=command.output_dir)

    return Container(
        ingest_phapdien_use_case=IngestPhapdienUseCase(
            dataset_loader=dataset_loader,
            chunking_policy=chunking_policy,
            law_repository=law_repository,
            command=command,
        )
    )


def build_import_vector_store_container(
    command: ImportParquetVectorStoreCommand,
    *,
    qdrant_url: str = "http://localhost:6333",
    qdrant_api_key: str | None = None,
    timeout_seconds: float = 120.0,
) -> ImportVectorStoreContainer:
    ImportParquetVectorStoreUseCase.validate_command(command)
    if timeout_seconds <= 0:
        raise ValueError("--timeout-seconds must be positive.")

    vector_store = QdrantVectorStore(
        collection_name=command.collection_name,
        url=qdrant_url,
        api_key=qdrant_api_key,
        timeout_seconds=timeout_seconds,
    )

    return ImportVectorStoreContainer(
        import_vector_store_use_case=ImportParquetVectorStoreUseCase(
            vector_shard_reader=ParquetVectorShardReader(),
            vector_store=vector_store,
            progress_reporter=ConsoleProgressReporter(),
            command=command,
        )
    )
