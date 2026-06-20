from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ai_legal_assistant.domain.entities.competition_submission import CompetitionQuestion


class JsonCompetitionQuestionSource:
    """Reads the competition's JSON array without changing the supplied question text."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> list[CompetitionQuestion]:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ValueError(f"Question file does not exist: {self.path}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"Question file is not valid JSON: {self.path}") from exc
        if not isinstance(payload, list):
            raise ValueError("Question file must contain a JSON array.")

        questions: list[CompetitionQuestion] = []
        seen_ids: set[int] = set()
        for index, row in enumerate(payload):
            question_id, question = self._parse_row(row, index)
            if question_id in seen_ids:
                raise ValueError(f"Duplicate question id in source file: {question_id}")
            seen_ids.add(question_id)
            questions.append(CompetitionQuestion(question_id=question_id, question=question))
        if not questions:
            raise ValueError("Question file must not be empty.")
        return questions

    @staticmethod
    def _parse_row(row: Any, index: int) -> tuple[int, str]:
        if not isinstance(row, dict):
            raise ValueError(f"questions[{index}] must be a JSON object.")
        question_id = row.get("id")
        question = row.get("question")
        if not isinstance(question_id, int) or isinstance(question_id, bool):
            raise ValueError(f"questions[{index}].id must be an integer.")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"questions[{index}].question must be a non-empty string.")
        return question_id, question
