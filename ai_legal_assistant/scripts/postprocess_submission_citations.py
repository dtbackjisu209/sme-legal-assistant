from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ai_legal_assistant.domain.entities.competition_submission import (
    LegalCitation,
    ResolvedLegalContext,
    SubmissionRecord,
)
from ai_legal_assistant.domain.services.submission_citation_selector import (
    SubmissionCitationSelector,
    SubmissionCitationSelectorConfig,
)
from ai_legal_assistant.infrastructure.persistence.json_submission_file import JsonSubmissionFile
from ai_legal_assistant.infrastructure.persistence.submission_artifact_writer import (
    SubmissionArtifactWriter,
)


@dataclass(frozen=True)
class _OrderedHit:
    score: float
    metadata: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rewrite relevant_docs/relevant_articles in an existing results.json "
            "without rerunning retrieval or answer generation."
        )
    )
    parser.add_argument("results", type=Path, help="Existing results.json")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for filtered results.json and submission.zip.",
    )
    parser.add_argument("--exact-citation-docs", type=int, default=1)
    parser.add_argument("--exact-citation-articles", type=int, default=1)
    parser.add_argument("--simple-citation-docs", type=int, default=2)
    parser.add_argument("--simple-citation-articles", type=int, default=2)
    parser.add_argument("--multi-citation-docs", type=int, default=3)
    parser.add_argument("--multi-citation-articles", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or args.results.parent / "filtered"
    selector = SubmissionCitationSelector(
        SubmissionCitationSelectorConfig(
            exact_lookup_max_docs=args.exact_citation_docs,
            exact_lookup_max_articles=args.exact_citation_articles,
            simple_max_docs=args.simple_citation_docs,
            simple_max_articles=args.simple_citation_articles,
            multi_issue_max_docs=args.multi_citation_docs,
            multi_issue_max_articles=args.multi_citation_articles,
        )
    )
    records = JsonSubmissionFile.load(args.results)
    filtered = filter_records(records, selector=selector)
    artifact = SubmissionArtifactWriter().write(records=filtered, output_dir=output_dir)
    before_docs = sum(len(record.relevant_docs) for record in records)
    before_articles = sum(len(record.relevant_articles) for record in records)
    after_docs = sum(len(record.relevant_docs) for record in filtered)
    after_articles = sum(len(record.relevant_articles) for record in filtered)
    print(f"Filtered {len(filtered)} records.")
    print(f"docs: {before_docs} -> {after_docs}")
    print(f"articles: {before_articles} -> {after_articles}")
    print(f"results.json: {artifact.results_path}")
    print(f"submission.zip: {artifact.zip_path}")
    return 0


def filter_records(
    records: list[SubmissionRecord],
    *,
    selector: SubmissionCitationSelector,
) -> list[SubmissionRecord]:
    return [_filter_record(record, selector=selector) for record in records]


def _filter_record(
    record: SubmissionRecord,
    *,
    selector: SubmissionCitationSelector,
) -> SubmissionRecord:
    citations = [_parse_article_entry(entry) for entry in record.relevant_articles]
    citations = [citation for citation in citations if citation is not None]
    hits = [
        _OrderedHit(score=float(len(citations) - index), metadata={})
        for index, _citation in enumerate(citations)
    ]
    contexts = [
        ResolvedLegalContext(text="", citation=citation)
        for citation in citations
    ]
    selection = selector.select(
        question=record.question,
        hits=hits,
        contexts=contexts,
    )
    max_docs, _max_articles = selector.limits_for_question(record.question)
    docs = _merge_docs(selection.relevant_docs, record.relevant_docs, max_docs=max_docs)
    return SubmissionRecord(
        question_id=record.question_id,
        question=record.question,
        answer=record.answer,
        relevant_docs=tuple(docs),
        relevant_articles=selection.relevant_articles,
    )


def _parse_article_entry(entry: str) -> LegalCitation | None:
    parts = tuple(part.strip() for part in entry.split("|"))
    if len(parts) != 3 or any(not part for part in parts):
        return None
    return LegalCitation(document_code=parts[0], document_title=parts[1], article=parts[2])


def _merge_docs(
    selected_docs: tuple[str, ...],
    original_docs: tuple[str, ...],
    *,
    max_docs: int,
) -> list[str]:
    docs: list[str] = []
    seen: set[str] = set()
    for entry in (*selected_docs, *original_docs):
        if entry in seen:
            continue
        seen.add(entry)
        docs.append(entry)
        if len(docs) >= max_docs:
            break
    return docs


if __name__ == "__main__":
    raise SystemExit(main())
