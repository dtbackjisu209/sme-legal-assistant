from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from ai_legal_assistant.application.dto.answer_generation_dto import (
    GenerateGroundedAnswerRequest,
)
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
from ai_legal_assistant.domain.services.submission_citation_selector import (
    SubmissionCitationSelection,
    SubmissionCitationSelector,
)
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
    citation_selector: SubmissionCitationSelector = field(
        default_factory=SubmissionCitationSelector
    )
    checkpoint: SubmissionCheckpointPort | None = None
    progress_callback: Callable[[int, int], None] | None = None

    def execute(self, command: GenerateSubmissionCommand) -> SubmissionArtifact:
        if command.retrieval_top_k <= 0:
            raise ValueError("retrieval_top_k must be positive.")
        if command.answer_batch_size <= 0:
            raise ValueError("answer_batch_size must be positive.")
        if command.checkpoint_interval <= 0:
            raise ValueError("checkpoint_interval must be positive.")

        questions = self.question_source.load()
        saved_records_by_id: dict[int, SubmissionRecord] = {}
        if self.checkpoint is not None:
            saved_records = self.checkpoint.load()
            self.validator.validate_partial(saved_records, questions)
            saved_records_by_id = {
                record.question_id: record for record in saved_records
            }
        records: list[SubmissionRecord] = []
        pending_answers: list[GenerateGroundedAnswerRequest] = []
        pending_citations: list[SubmissionCitationSelection] = []
        unsaved_record_count = 0

        def persist_checkpoint(*, force: bool = False) -> None:
            nonlocal unsaved_record_count
            if self.checkpoint is None:
                return
            if force or unsaved_record_count >= command.checkpoint_interval:
                self.checkpoint.save(records)
                unsaved_record_count = 0

        def generate_pending_answers() -> None:
            nonlocal unsaved_record_count
            if not pending_answers:
                return
            requests = tuple(pending_answers)
            citation_selections = tuple(pending_citations)
            pending_answers.clear()
            pending_citations.clear()
            answers = self.answer_generator.generate_batch(requests=requests)
            if len(answers) != len(requests):
                raise ValueError("answer generator returned an unexpected batch size.")
            if len(citation_selections) != len(requests):
                raise ValueError("citation selector returned an unexpected batch size.")
            for request, answer, citation_selection in zip(
                requests,
                answers,
                citation_selections,
                strict=True,
            ):
                records.append(
                    SubmissionRecord(
                        question_id=request.question.question_id,
                        question=request.question.question,
                        answer=answer.strip(),
                        relevant_docs=citation_selection.relevant_docs,
                        relevant_articles=citation_selection.relevant_articles,
                    )
                )
                unsaved_record_count += 1
            persist_checkpoint()
            if self.progress_callback is not None:
                self.progress_callback(len(records), len(questions))

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
            citation_selection = self.citation_selector.select(
                question=question.question,
                hits=hits,
                contexts=contexts,
            )
            pending_answers.append(
                GenerateGroundedAnswerRequest(
                    question=question,
                    contexts=tuple(contexts),
                )
            )
            pending_citations.append(citation_selection)
            if len(pending_answers) >= command.answer_batch_size:
                generate_pending_answers()

        generate_pending_answers()
        question_order = {
            question.question_id: index for index, question in enumerate(questions)
        }
        records.sort(key=lambda record: question_order[record.question_id])
        persist_checkpoint(force=True)

        self.validator.validate(records, questions)
        artifact = self.artifact_writer.write(records=records, output_dir=command.output_dir)
        if self.checkpoint is not None:
            self.checkpoint.clear()
        return artifact
