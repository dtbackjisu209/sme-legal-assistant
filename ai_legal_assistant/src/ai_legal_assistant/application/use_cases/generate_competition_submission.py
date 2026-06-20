from __future__ import annotations

from dataclasses import dataclass

from ai_legal_assistant.application.dto.retrieval_dto import RetrieveLegalContextQuery
from ai_legal_assistant.application.dto.submission_dto import (
    GenerateSubmissionCommand,
    SubmissionArtifact,
)
from ai_legal_assistant.application.ports.competition_question_source_port import (
    CompetitionQuestionSourcePort,
)
from ai_legal_assistant.application.ports.legal_answer_generator_port import (
    LegalAnswerGeneratorPort,
)
from ai_legal_assistant.application.ports.legal_citation_resolver_port import (
    LegalCitationResolverPort,
)
from ai_legal_assistant.application.ports.retriever_port import LegalContextRetrieverPort
from ai_legal_assistant.application.ports.submission_artifact_port import SubmissionArtifactPort
from ai_legal_assistant.application.ports.submission_checkpoint_port import SubmissionCheckpointPort
from ai_legal_assistant.domain.entities.competition_submission import SubmissionRecord
from ai_legal_assistant.domain.services.submission_validator import SubmissionValidator


@dataclass
class GenerateCompetitionSubmissionUseCase:
    """Generate every prediction, validate it, then atomically create the upload artifact."""

    question_source: CompetitionQuestionSourcePort
    retriever: LegalContextRetrieverPort
    citation_resolver: LegalCitationResolverPort
    answer_generator: LegalAnswerGeneratorPort
    validator: SubmissionValidator
    artifact_writer: SubmissionArtifactPort
    checkpoint: SubmissionCheckpointPort | None = None

    def execute(self, command: GenerateSubmissionCommand) -> SubmissionArtifact:
        if command.retrieval_top_k <= 0:
            raise ValueError("retrieval_top_k must be positive.")

        questions = self.question_source.load()
        saved_records_by_id: dict[int, SubmissionRecord] = {}
        if self.checkpoint is not None:
            saved_records = self.checkpoint.load()
            self.validator.validate_partial(saved_records, questions)
            saved_records_by_id = {
                record.question_id: record for record in saved_records
            }
        records: list[SubmissionRecord] = []
        for question in questions:
            saved_record = saved_records_by_id.get(question.question_id)
            if saved_record is not None:
                records.append(saved_record)
                continue
            hits = self.retriever.execute(
                RetrieveLegalContextQuery(
                    query=question.question,
                    top_k=command.retrieval_top_k,
                )
            )
            contexts = self.citation_resolver.resolve(hits)
            answer = self.answer_generator.generate(question=question, contexts=contexts)
            citations = tuple(
                context.citation for context in contexts if context.citation is not None
            )
            records.append(
                SubmissionRecord(
                    question_id=question.question_id,
                    question=question.question,
                    answer=answer.strip(),
                    relevant_docs=tuple(
                        dict.fromkeys(citation.document_entry for citation in citations)
                    ),
                    relevant_articles=tuple(
                        dict.fromkeys(citation.article_entry for citation in citations)
                    ),
                )
            )
            if self.checkpoint is not None:
                self.checkpoint.save(records)

        self.validator.validate(records, questions)
        artifact = self.artifact_writer.write(records=records, output_dir=command.output_dir)
        if self.checkpoint is not None:
            self.checkpoint.clear()
        return artifact
