from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ai_legal_assistant.domain.entities.bm25_index import SearchableLawChunk
from ai_legal_assistant.domain.services.bm25_index_builder import BM25IndexBuilder
from ai_legal_assistant.infrastructure.nlp.vietnamese_legal_tokenizer import (
    VietnameseLegalTokenizer,
    VietnameseLegalTokenizerConfig,
)
from ai_legal_assistant.infrastructure.search.bm25_index_store import PickleBM25IndexStore
from ai_legal_assistant.infrastructure.search.bm25_search import BM25Search


class VietnameseLegalTokenizerTest(unittest.TestCase):
    def test_repairs_mojibake_and_adds_legal_phrase_tokens(self) -> None:
        tokenizer = VietnameseLegalTokenizer(VietnameseLegalTokenizerConfig(backend="regex"))
        mojibake = "Luật quy định về".encode("utf-8").decode("latin1")

        tokens = tokenizer.tokenize(f"{mojibake} vốn điều lệ và mã số thuế.")

        self.assertIn("luật", tokens)
        self.assertIn("vốn_điều_lệ", tokens)
        self.assertIn("mã_số_thuế", tokens)


class BM25SearchTest(unittest.TestCase):
    def test_search_uses_postings_for_legal_phrase_tokens(self) -> None:
        tokenizer = VietnameseLegalTokenizer(VietnameseLegalTokenizerConfig(backend="regex"))
        index = BM25IndexBuilder(tokenizer=tokenizer).build(
            [
                SearchableLawChunk(
                    chunk_id="chunk-1",
                    text="Công ty phải góp đủ vốn điều lệ trong thời hạn luật định.",
                    metadata={"article": "1"},
                ),
                SearchableLawChunk(
                    chunk_id="chunk-2",
                    text="Doanh nghiệp phải đăng ký mã số thuế trước khi xuất hóa đơn.",
                    metadata={"article": "2"},
                ),
            ]
        )

        results = BM25Search(index=index, tokenizer=tokenizer).search("không góp đủ vốn điều lệ", top_k=1)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].chunk_id, "chunk-1")
        self.assertIn("vốn_điều_lệ", index.postings)

    def test_search_loads_tokenizer_config_from_infrastructure_payload(self) -> None:
        tokenizer = VietnameseLegalTokenizer(VietnameseLegalTokenizerConfig(backend="regex"))
        index = BM25IndexBuilder(tokenizer=tokenizer).build(
            [
                SearchableLawChunk(
                    chunk_id="chunk-1",
                    text="Công ty phải góp đủ vốn điều lệ.",
                    metadata={},
                )
            ]
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            index_path = Path(temp_dir) / "bm25_index.pkl"
            PickleBM25IndexStore(index_path, tokenizer_config=tokenizer.export_config()).save(index)

            results = BM25Search.from_path(index_path).search("vốn điều lệ", top_k=1)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].chunk_id, "chunk-1")


if __name__ == "__main__":
    unittest.main()
