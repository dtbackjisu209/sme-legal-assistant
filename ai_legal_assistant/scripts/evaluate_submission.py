from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ai_legal_assistant.domain.entities.competition_submission import SubmissionRecord
from ai_legal_assistant.domain.services.submission_metrics import (
    SubmissionEvaluationResult,
    SubmissionMetricsCalculator,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate submission relevant_docs/relevant_articles against a local "
            "gold file with leaderboard-style macro precision, recall, and F2."
        )
    )
    parser.add_argument("predictions", type=Path, help="Path to predicted results.json")
    parser.add_argument("--gold", type=Path, required=True, help="Path to gold labels JSON")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = SubmissionMetricsCalculator().evaluate(
        predictions=_load_label_records(args.predictions),
        gold=_load_label_records(args.gold),
    )
    if args.json:
        print(json.dumps(_to_mapping(result), ensure_ascii=False, indent=2))
    else:
        _print_table(result)
    return 0


def _load_label_records(path: Path) -> list[SubmissionRecord]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"label file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"label file is not valid JSON: {path}") from exc
    if not isinstance(payload, list):
        raise ValueError("label file must contain a JSON array.")

    records: list[SubmissionRecord] = []
    for index, row in enumerate(payload):
        if not isinstance(row, dict):
            raise ValueError(f"records[{index}] must be a JSON object.")
        question_id = row.get("id")
        if not isinstance(question_id, int) or isinstance(question_id, bool):
            raise ValueError(f"records[{index}].id must be an integer.")
        relevant_docs = _string_list(row.get("relevant_docs"), f"records[{index}].relevant_docs")
        relevant_articles = _string_list(
            row.get("relevant_articles"),
            f"records[{index}].relevant_articles",
        )
        records.append(
            SubmissionRecord(
                question_id=question_id,
                question=str(row.get("question") or ""),
                answer=str(row.get("answer") or ""),
                relevant_docs=tuple(relevant_docs),
                relevant_articles=tuple(relevant_articles),
            )
        )
    return records


def _string_list(value: Any, field: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be an array.")
    if not all(isinstance(item, str) for item in value):
        raise ValueError(f"{field} must contain only strings.")
    return value


def _to_mapping(result: SubmissionEvaluationResult) -> dict[str, Any]:
    return {
        "case_count": result.case_count,
        "docs": {
            "precision": result.docs.precision,
            "recall": result.docs.recall,
            "f2": result.docs.f2,
        },
        "articles": {
            "precision": result.articles.precision,
            "recall": result.articles.recall,
            "f2": result.articles.f2,
        },
    }


def _print_table(result: SubmissionEvaluationResult) -> None:
    print(f"Cases: {result.case_count}")
    print("field      precision  recall     f2")
    print(
        f"docs       {result.docs.precision:9.4f}  "
        f"{result.docs.recall:6.4f}  {result.docs.f2:6.4f}"
    )
    print(
        f"articles   {result.articles.precision:9.4f}  "
        f"{result.articles.recall:6.4f}  {result.articles.f2:6.4f}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
