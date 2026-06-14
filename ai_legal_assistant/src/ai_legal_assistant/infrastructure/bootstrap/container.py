from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ai_legal_assistant.application.dto.ingestion_dto import IngestPhapdienCommand
from ai_legal_assistant.application.use_cases.ingest_phapdien import IngestPhapdienUseCase
from ai_legal_assistant.domain.services.legal_chunking_policy import LegalChunkingPolicy
from ai_legal_assistant.infrastructure.datasets.huggingface_phapdien_loader import HuggingFacePhapdienLoader
from ai_legal_assistant.infrastructure.persistence.jsonl_law_repository import JsonlLawRepository


@dataclass(frozen=True)
class Container:
    ingest_phapdien_use_case: IngestPhapdienUseCase


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
