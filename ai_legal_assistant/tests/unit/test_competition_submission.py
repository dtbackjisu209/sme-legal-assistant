from __future__ import annotations

import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ai_legal_assistant.application.dto.retrieval_dto import DenseSearchResult
from ai_legal_assistant.application.dto.submission_dto import GenerateSubmissionCommand
from ai_legal_assistant.application.use_cases.generate_competition_submission import (
    GenerateCompetitionSubmissionUseCase,
)
from ai_legal_assistant.domain.entities.competition_submission import (
    CompetitionQuestion,
    LegalCitation,
    ResolvedLegalContext,
    SubmissionRecord,
)
from ai_legal_assistant.domain.services.submission_validator import (
    SubmissionValidationError,
    SubmissionValidator,
)
from ai_legal_assistant.infrastructure.citations.jsonl_legal_citation_resolver import (
    JsonlLegalCitationResolver,
)
from ai_legal_assistant.infrastructure.persistence.json_submission_file import JsonSubmissionFile
from ai_legal_assistant.infrastructure.persistence.submission_artifact_writer import (
    SubmissionArtifactWriter,
)


class StaticQuestionSource:
    def __init__(self, questions: list[CompetitionQuestion]) -> None:
        self.questions = questions

    def load(self) -> list[CompetitionQuestion]:
        return self.questions


class StaticRetriever:
    def execute(self, request):
        return [
            DenseSearchResult(
                chunk_id="chunk-1",
                score=1.0,
                text="Nội dung Điều 4.",
                metadata={"article_id": "article-1"},
            )
        ]


class StaticCitationResolver:
    def resolve(self, hits):
        return [
            ResolvedLegalContext(
                text=hits[0].text,
                citation=LegalCitation(
                    document_code="04/2017/QH14",
                    document_title="Luật 04/2017/QH14 Luật Hỗ trợ doanh nghiệp nhỏ và vừa",
                    article="Điều 4",
                ),
            )
        ]


class StaticAnswerGenerator:
    def __init__(self) -> None:
        self.batch_question_ids: list[list[int]] = []

    def generate(self, *, question, contexts):
        return "Doanh nghiệp được hỗ trợ theo Điều 4 của Luật 04/2017/QH14."


    def generate_batch(self, *, requests):
        self.batch_question_ids.append(
            [request.question.question_id for request in requests]
        )
        return tuple(
            self.generate(question=request.question, contexts=request.contexts)
            for request in requests
        )


class RecordingCheckpoint:
    def __init__(self) -> None:
        self.saved_sizes: list[int] = []
        self.cleared = False

    def load(self):
        return []

    def save(self, records):
        self.saved_sizes.append(len(records))

    def clear(self):
        self.cleared = True


class CompetitionSubmissionTest(unittest.TestCase):
    def test_pipeline_writes_flat_valid_zip(self) -> None:
        questions = [CompetitionQuestion(question_id=1, question="Điều kiện hỗ trợ là gì?")]
        with tempfile.TemporaryDirectory() as directory:
            use_case = GenerateCompetitionSubmissionUseCase(
                question_source=StaticQuestionSource(questions),
                retriever=StaticRetriever(),
                citation_resolver=StaticCitationResolver(),
                answer_generator=StaticAnswerGenerator(),
                validator=SubmissionValidator(),
                artifact_writer=SubmissionArtifactWriter(),
            )

            artifact = use_case.execute(
                GenerateSubmissionCommand(output_dir=Path(directory), retrieval_top_k=3)
            )

            self.assertEqual(artifact.record_count, 1)
            records = JsonSubmissionFile.load(artifact.results_path)
            self.assertEqual(records[0].relevant_docs, (
                "04/2017/QH14|Luật 04/2017/QH14 Luật Hỗ trợ doanh nghiệp nhỏ và vừa",
            ))
            with zipfile.ZipFile(artifact.zip_path) as archive:
                self.assertEqual(archive.namelist(), ["results.json"])
                self.assertEqual(archive.read("results.json"), artifact.results_path.read_bytes())

    def test_pipeline_batches_answers_and_checkpoints_by_interval(self) -> None:
        questions = [
            CompetitionQuestion(question_id=index, question=f"Question {index}")
            for index in range(1, 6)
        ]
        answer_generator = StaticAnswerGenerator()
        checkpoint = RecordingCheckpoint()
        with tempfile.TemporaryDirectory() as directory:
            use_case = GenerateCompetitionSubmissionUseCase(
                question_source=StaticQuestionSource(questions),
                retriever=StaticRetriever(),
                citation_resolver=StaticCitationResolver(),
                answer_generator=answer_generator,
                validator=SubmissionValidator(),
                artifact_writer=SubmissionArtifactWriter(),
                checkpoint=checkpoint,
            )

            artifact = use_case.execute(
                GenerateSubmissionCommand(
                    output_dir=Path(directory),
                    retrieval_top_k=3,
                    answer_batch_size=2,
                    checkpoint_interval=3,
                )
            )

        self.assertEqual(artifact.record_count, 5)
        self.assertEqual(answer_generator.batch_question_ids, [[1, 2], [3, 4], [5]])
        self.assertEqual(checkpoint.saved_sizes, [4, 5])
        self.assertTrue(checkpoint.cleared)

    def test_validator_rejects_missing_test_question(self) -> None:
        expected = [
            CompetitionQuestion(question_id=1, question="Câu hỏi 1"),
            CompetitionQuestion(question_id=2, question="Câu hỏi 2"),
        ]
        records = [
            SubmissionRecord(
                question_id=1,
                question="Câu hỏi 1",
                answer="Câu trả lời",
                relevant_docs=(),
                relevant_articles=(),
            )
        ]

        with self.assertRaisesRegex(SubmissionValidationError, "missing ids"):
            SubmissionValidator().validate(records, expected)

    def test_citation_resolver_restores_title_missing_from_later_article_note(self) -> None:
        rows = [
            {
                "article_id": "one",
                "source_note_text": (
                    "(Điều 1 Luật số 32/2004/QH11 An ninh Quốc gia ngày 03/12/2004 "
                    "của Quốc hội)"
                ),
            },
            {
                "article_id": "two",
                "source_note_text": "(Điều 2 Luật số 32/2004/QH11, có hiệu lực thi hành)",
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            corpus_path = Path(directory) / "law_articles.jsonl"
            corpus_path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )
            resolver = JsonlLegalCitationResolver(corpus_path)
            contexts = resolver.resolve(
                [
                    DenseSearchResult(
                        chunk_id="chunk-2",
                        score=0.9,
                        text="Nội dung Điều 2",
                        metadata={"article_id": "two"},
                    )
                ]
            )

        self.assertEqual(contexts[0].citation.document_code, "32/2004/QH11")
        self.assertEqual(
            contexts[0].citation.document_title,
            "Luật 32/2004/QH11 An ninh Quốc gia",
        )
        self.assertEqual(contexts[0].citation.article, "Điều 2")


if __name__ == "__main__":
    unittest.main()
