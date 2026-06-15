from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ai_legal_assistant.application.dto.retrieval_dto import BuildBM25IndexCommand
from ai_legal_assistant.application.use_cases.build_bm25_index import BuildBM25IndexUseCase
from ai_legal_assistant.domain.services.bm25_index_builder import BM25IndexBuilder
from ai_legal_assistant.infrastructure.nlp.vietnamese_legal_tokenizer import (
    DEFAULT_LEGAL_PHRASES,
    VietnameseLegalTokenizer,
    VietnameseLegalTokenizerConfig,
)
from ai_legal_assistant.infrastructure.persistence.jsonl_law_chunk_reader import JsonlLawChunkReader
from ai_legal_assistant.infrastructure.search.bm25_index_store import PickleBM25IndexStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Vietnamese legal BM25 inverted index.")
    parser.add_argument(
        "--chunks-path",
        type=Path,
        default=ROOT_DIR / "data" / "processed" / "law_chunks.jsonl",
        help="Path to law_chunks.jsonl.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=ROOT_DIR / "data" / "indexes" / "bm25_index.pkl",
        help="Path to write bm25_index.pkl.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional number of chunks to index.")
    parser.add_argument("--k1", type=float, default=1.5, help="BM25 k1 parameter.")
    parser.add_argument("--b", type=float, default=0.75, help="BM25 b parameter.")
    parser.add_argument(
        "--tokenizer",
        choices=("auto", "underthesea", "pyvi", "regex"),
        default="auto",
        help="Tokenizer backend. auto tries underthesea, then pyvi, then regex.",
    )
    parser.add_argument(
        "--phrase-file",
        type=Path,
        default=None,
        help="Optional UTF-8 text file with one legal phrase per line.",
    )
    parser.add_argument(
        "--phrase-boost",
        type=int,
        default=1,
        help="Additional occurrences added for each matched legal phrase token.",
    )
    parser.add_argument(
        "--no-repair-mojibake",
        action="store_true",
        help="Disable mojibake repair for mis-decoded Vietnamese text.",
    )
    return parser.parse_args()


def load_phrases(path: Path | None) -> tuple[str, ...]:
    if path is None:
        return DEFAULT_LEGAL_PHRASES
    with path.open("r", encoding="utf-8") as handle:
        custom_phrases = tuple(line.strip() for line in handle if line.strip() and not line.startswith("#"))
    return tuple(dict.fromkeys((*DEFAULT_LEGAL_PHRASES, *custom_phrases)))


def main() -> int:
    args = parse_args()
    command = BuildBM25IndexCommand(
        chunks_path=args.chunks_path,
        output_path=args.output_path,
        limit=args.limit,
    )
    tokenizer = VietnameseLegalTokenizer(
        VietnameseLegalTokenizerConfig(
            backend=args.tokenizer,
            legal_phrases=load_phrases(args.phrase_file),
            phrase_boost=args.phrase_boost,
            repair_mojibake=not args.no_repair_mojibake,
        )
    )
    use_case = BuildBM25IndexUseCase(
        chunk_reader=JsonlLawChunkReader(command.chunks_path),
        index_store=PickleBM25IndexStore(command.output_path, tokenizer_config=tokenizer.export_config()),
        index_builder=BM25IndexBuilder(tokenizer=tokenizer, k1=args.k1, b=args.b),
    )
    result = use_case.execute(command)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
