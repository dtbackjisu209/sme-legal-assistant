from __future__ import annotations

import os
import tempfile
import zipfile
from collections.abc import Sequence
from pathlib import Path

from ai_legal_assistant.application.dto.submission_dto import SubmissionArtifact
from ai_legal_assistant.domain.entities.competition_submission import SubmissionRecord
from ai_legal_assistant.infrastructure.persistence.json_submission_file import JsonSubmissionFile


class SubmissionArtifactWriter:
    """Writes an atomic ``results.json`` and a flat, dashboard-compatible ZIP file."""

    RESULTS_FILENAME = "results.json"
    ZIP_FILENAME = "submission.zip"

    def write(
        self,
        *,
        records: Sequence[SubmissionRecord],
        output_dir: Path,
    ) -> SubmissionArtifact:
        output_dir.mkdir(parents=True, exist_ok=True)
        results_path = output_dir / self.RESULTS_FILENAME
        zip_path = output_dir / self.ZIP_FILENAME
        self._atomic_write_text(
            results_path,
            JsonSubmissionFile.dump(list(records)),
        )
        self._atomic_write_zip(zip_path, results_path)
        self._verify_zip(zip_path)
        return SubmissionArtifact(
            record_count=len(records),
            results_path=results_path,
            zip_path=zip_path,
        )

    @staticmethod
    def _atomic_write_text(path: Path, content: str) -> None:
        handle = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            delete=False,
        )
        try:
            with handle:
                handle.write(content)
            os.replace(handle.name, path)
        except Exception:
            Path(handle.name).unlink(missing_ok=True)
            raise

    @staticmethod
    def _atomic_write_zip(zip_path: Path, results_path: Path) -> None:
        handle = tempfile.NamedTemporaryFile(dir=zip_path.parent, delete=False)
        handle.close()
        try:
            with zipfile.ZipFile(handle.name, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.write(results_path, arcname=SubmissionArtifactWriter.RESULTS_FILENAME)
            os.replace(handle.name, zip_path)
        except Exception:
            Path(handle.name).unlink(missing_ok=True)
            raise

    @staticmethod
    def _verify_zip(zip_path: Path) -> None:
        with zipfile.ZipFile(zip_path) as archive:
            if archive.namelist() != [SubmissionArtifactWriter.RESULTS_FILENAME]:
                raise ValueError(
                    "Submission ZIP must contain only results.json at its root."
                )
