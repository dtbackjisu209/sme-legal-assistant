from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ai_legal_assistant.domain.entities.competition_submission import SubmissionRecord


class JsonSubmissionFile:
    """Serializes and deserializes the dashboard's ``results.json`` schema."""

    @staticmethod
    def load(path: Path) -> list[SubmissionRecord]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ValueError(f"results.json does not exist: {path}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"results.json is not valid JSON: {path}") from exc
        if not isinstance(payload, list):
            raise ValueError("results.json must contain a JSON array.")
        return [JsonSubmissionFile._parse_record(row, index) for index, row in enumerate(payload)]

    @staticmethod
    def dump(records: list[SubmissionRecord]) -> str:
        return json.dumps(
            [record.to_mapping() for record in records],
            ensure_ascii=False,
            indent=2,
        ) + "\n"

    @staticmethod
    def _parse_record(row: Any, index: int) -> SubmissionRecord:
        if not isinstance(row, dict):
            raise ValueError(f"results[{index}] must be a JSON object.")
        try:
            question_id = row["id"]
            question = row["question"]
            answer = row["answer"]
            relevant_docs = row["relevant_docs"]
            relevant_articles = row["relevant_articles"]
        except KeyError as exc:
            raise ValueError(f"results[{index}] is missing required field: {exc.args[0]}") from exc
        if not isinstance(relevant_docs, list) or not isinstance(relevant_articles, list):
            raise ValueError(
                f"results[{index}].relevant_docs and relevant_articles must be arrays."
            )
        return SubmissionRecord(
            question_id=question_id,
            question=question,
            answer=answer,
            relevant_docs=tuple(relevant_docs),
            relevant_articles=tuple(relevant_articles),
        )
