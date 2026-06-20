from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ai_legal_assistant.domain.services.submission_validator import SubmissionValidator
from ai_legal_assistant.infrastructure.persistence.json_competition_question_source import (
    JsonCompetitionQuestionSource,
)
from ai_legal_assistant.infrastructure.persistence.json_submission_file import JsonSubmissionFile
from ai_legal_assistant.infrastructure.persistence.submission_artifact_writer import (
    SubmissionArtifactWriter,
)


DEFAULT_QUESTIONS_PATH = ROOT_DIR / "data" / "raw" / "R2AIStage1DATA (1).json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate results.json and, optionally, its upload ZIP before using a submission quota."
    )
    parser.add_argument("results", type=Path, help="Path to results.json")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS_PATH)
    parser.add_argument("--zip", dest="zip_path", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = JsonSubmissionFile.load(args.results)
    questions = JsonCompetitionQuestionSource(args.questions).load()
    SubmissionValidator().validate(records, questions)
    if args.zip_path is not None:
        with zipfile.ZipFile(args.zip_path) as archive:
            if archive.namelist() != [SubmissionArtifactWriter.RESULTS_FILENAME]:
                raise ValueError("ZIP must contain only results.json at its root.")
            uploaded_content = archive.read(SubmissionArtifactWriter.RESULTS_FILENAME)
        if uploaded_content != args.results.read_bytes():
            raise ValueError("ZIP's results.json differs from the file that was validated.")
    print(f"Valid submission: {len(records)} results.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
