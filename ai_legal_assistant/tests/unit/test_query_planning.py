from __future__ import annotations

import json
import sys
import unicodedata
import unittest
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ai_legal_assistant.application.use_cases.rewrite_legal_query import (
    BuildLegalQueryPlanUseCase,
)
from ai_legal_assistant.domain.entities.legal_query import LegalQuery
from ai_legal_assistant.domain.entities.query_plan import (
    ExpansionProposal,
    QueryAnalysis,
    QueryType,
    QueryVariantKind,
    TemporalScope,
    WeightedQuery,
)
from ai_legal_assistant.domain.services.query_expansion_policy import QueryExpansionPolicy
from ai_legal_assistant.infrastructure.nlp.vietnamese_legal_query_normalizer import (
    VietnameseLegalQueryNormalizer,
)
from ai_legal_assistant.infrastructure.query_expansion.llm_legal_query_planner import (
    LLMLegalQueryPlanner,
)


class FakeLLM:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.calls: list[tuple[str, str]] = []

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        forbidden_phrases: tuple[str, ...] = (),
    ) -> str:
        self.calls.append((system_prompt, user_prompt))
        return json.dumps(self.payload, ensure_ascii=False)


class SequenceLLM:
    def __init__(self, payloads: list[dict[str, Any]]) -> None:
        self.payloads = payloads
        self.calls = 0

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        forbidden_phrases: tuple[str, ...] = (),
    ) -> str:
        payload = self.payloads[self.calls]
        self.calls += 1
        return json.dumps(payload, ensure_ascii=False)


def base_payload(
    *,
    intent: str = "deadline",
    query_type: str = "legal_concept",
) -> dict[str, Any]:
    return {
        "analysis": {
            "intent": intent,
            "query_type": query_type,
            "legal_domain": "enterprise",
            "entities": [],
            "company_type": None,
            "article_number": None,
            "clause_number": None,
            "document_number": None,
            "temporal_scope": "current",
            "must_terms": [],
        },
        "semantic_variants": [],
        "subqueries": [],
        "lexical_terms": [],
    }


def planner_with_payload(payload: dict[str, Any]) -> tuple[BuildLegalQueryPlanUseCase, FakeLLM]:
    llm = FakeLLM(payload)
    use_case = BuildLegalQueryPlanUseCase(
        normalizer=VietnameseLegalQueryNormalizer(),
        planner=LLMLegalQueryPlanner(llm=llm),
        policy=QueryExpansionPolicy(),
    )
    return use_case, llm


class VietnameseLegalQueryNormalizerTest(unittest.TestCase):
    def test_normalizes_without_changing_negation(self) -> None:
        raw = unicodedata.normalize(
            "NFD",
            "  Công ty TNHH không thực hiện khoản2 điều47 thì sao?  ",
        )

        normalized = VietnameseLegalQueryNormalizer().normalize(LegalQuery(raw))

        self.assertEqual(
            normalized.text,
            "Công ty trách nhiệm hữu hạn không thực hiện Khoản 2 Điều 47 thì sao?",
        )

    def test_normalizes_document_number(self) -> None:
        normalized = VietnameseLegalQueryNormalizer().normalize(
            LegalQuery("Nghị định 123 / 2020 / NĐ - CP quy định gì?")
        )

        self.assertEqual(normalized.text, "Nghị định 123/2020/NĐ-CP quy định gì?")


class LLMQueryPlanningTest(unittest.TestCase):
    def test_rejected_expansion_falls_back_to_original_query(self) -> None:
        analysis_payload = base_payload(query_type="ambiguous")
        rejected_expansion = {
            "semantic_variants": [],
            "subqueries": [],
            "lexical_terms": [],
        }
        llm = SequenceLLM([analysis_payload, rejected_expansion])
        planner = BuildLegalQueryPlanUseCase(
            normalizer=VietnameseLegalQueryNormalizer(),
            planner=LLMLegalQueryPlanner(llm=llm),
            policy=QueryExpansionPolicy(),
        )

        plan = planner.execute("capital contribution deadline")

        self.assertEqual(len(plan.semantic_queries), 1)
        self.assertEqual(plan.semantic_queries[0].kind, QueryVariantKind.ORIGINAL)
        self.assertIn("fell back", plan.warnings[0])
        self.assertEqual(llm.calls, 2)

    def test_ambiguous_query_uses_analysis_and_expansion_llm_calls(self) -> None:
        payload = base_payload(query_type="ambiguous")
        payload["analysis"]["entities"] = [
            {"value": "vốn điều lệ", "evidence": "vốn điều lệ"}
        ]
        payload["analysis"]["must_terms"] = [
            {"value": "vốn điều lệ", "evidence": "vốn điều lệ"}
        ]
        payload["semantic_variants"] = [
            {
                "text": "Thời hạn góp vốn của công ty trách nhiệm hữu hạn một thành viên",
                "kind": "scope",
                "reason": "branch_one_member",
            },
            {
                "text": "Thời hạn góp vốn của công ty trách nhiệm hữu hạn hai thành viên trở lên",
                "kind": "scope",
                "reason": "branch_two_members",
            },
            {
                "text": "Thời hạn thanh toán cổ phần của công ty cổ phần",
                "kind": "scope",
                "reason": "branch_joint_stock",
            },
        ]
        planner, llm = planner_with_payload(payload)

        plan = planner.execute("Thời hạn góp đủ vốn điều lệ là bao lâu?")

        self.assertEqual(plan.analysis.query_type, QueryType.AMBIGUOUS)
        self.assertEqual(len(plan.semantic_queries), 4)
        self.assertEqual(len(llm.calls), 2)
        self.assertIn("Thời hạn góp đủ vốn điều lệ", llm.calls[0][1])
        self.assertIn("ĐÚNG 3 phần tử", llm.calls[1][0])
        self.assertIn("KHÔNG ĐƯỢC SAO CHÉP QUERY GỐC", llm.calls[1][1])

    def test_incomplete_ambiguous_expansion_uses_one_llm_repair_call(self) -> None:
        analysis_payload = base_payload(query_type="ambiguous")
        analysis_payload["analysis"]["entities"] = [
            {"value": "vốn điều lệ", "evidence": "vốn điều lệ"}
        ]
        analysis_payload["analysis"]["must_terms"] = [
            {"value": "vốn điều lệ", "evidence": "vốn điều lệ"}
        ]
        repair_payload = {
            "semantic_variants": [
                {"text": "Vốn điều lệ công ty trách nhiệm hữu hạn một thành viên", "kind": "scope", "reason": "one"},
                {"text": "Vốn điều lệ công ty trách nhiệm hữu hạn từ hai thành viên", "kind": "scope", "reason": "two"},
                {"text": "Cổ phần đăng ký mua của công ty cổ phần", "kind": "scope", "reason": "three"},
            ],
            "subqueries": [],
            "lexical_terms": ["vốn điều lệ", "văn điều lệ"],
        }
        llm = SequenceLLM([analysis_payload, repair_payload])
        planner = BuildLegalQueryPlanUseCase(
            normalizer=VietnameseLegalQueryNormalizer(),
            planner=LLMLegalQueryPlanner(llm=llm),
            policy=QueryExpansionPolicy(),
        )

        plan = planner.execute("Thời hạn góp đủ vốn điều lệ là bao lâu?")

        self.assertEqual(llm.calls, 2)
        self.assertEqual(len(plan.semantic_queries), 4)
        self.assertNotIn("văn điều lệ", plan.lexical_terms)

    def test_explicit_company_type_rejects_conflicting_llm_variant(self) -> None:
        payload = base_payload()
        payload["analysis"]["entities"] = [{"value": "góp vốn", "evidence": "góp đủ vốn"}]
        payload["analysis"]["company_type"] = {
            "value": "limited_liability_one_member",
            "evidence": "trách nhiệm hữu hạn một thành viên",
        }
        payload["analysis"]["must_terms"] = [
            {"value": "góp vốn", "evidence": "góp đủ vốn"}
        ]
        payload["semantic_variants"] = [
            {
                "text": "Thời hạn chủ sở hữu công ty trách nhiệm hữu hạn một thành viên góp đủ vốn",
                "kind": "semantic",
                "reason": "valid_rewrite",
            },
            {
                "text": "Thời hạn cổ đông công ty cổ phần thanh toán cổ phần",
                "kind": "scope",
                "reason": "conflicting_company_type",
            },
        ]
        planner, _ = planner_with_payload(payload)

        plan = planner.execute("Công ty TNHH một thành viên phải góp đủ vốn trong bao lâu?")

        self.assertEqual(plan.analysis.company_type, "limited_liability_one_member")
        self.assertEqual(len(plan.semantic_queries), 2)
        self.assertNotIn("công ty cổ phần", plan.semantic_queries[1].text)

    def test_exact_lookup_discards_llm_expansion(self) -> None:
        payload = base_payload(query_type="exact_lookup", intent="unknown")
        payload["analysis"]["entities"] = [{"value": "góp vốn", "evidence": "góp vốn"}]
        payload["analysis"]["article_number"] = {"value": "47", "evidence": "Điều 47"}
        payload["analysis"]["clause_number"] = {"value": "2", "evidence": "Khoản 2"}
        payload["analysis"]["must_terms"] = [
            {"value": "góp vốn", "evidence": "góp vốn"}
        ]
        planner, _ = planner_with_payload(payload)

        plan = planner.execute("Khoản 2 Điều 47 quy định gì về góp vốn?")

        self.assertEqual(len(plan.semantic_queries), 1)
        self.assertEqual(plan.analysis.article_number, "47")
        self.assertEqual(plan.analysis.clause_number, "2")

    def test_negation_drift_is_rejected(self) -> None:
        payload = base_payload(query_type="legal_situation", intent="penalty")
        payload["analysis"]["entities"] = [{"value": "góp vốn", "evidence": "góp đủ vốn"}]
        payload["analysis"]["must_terms"] = [
            {"value": "góp vốn", "evidence": "góp đủ vốn"}
        ]
        payload["semantic_variants"] = [
            {
                "text": "Xử phạt khi công ty góp đủ vốn điều lệ",
                "kind": "semantic",
                "reason": "lost_negation",
            },
            {
                "text": "Xử phạt khi công ty chưa góp đủ vốn điều lệ",
                "kind": "semantic",
                "reason": "preserved_negation",
            },
        ]
        planner, _ = planner_with_payload(payload)

        plan = planner.execute("Công ty tôi chưa góp đủ vốn và bị xử phạt thế nào?")

        self.assertEqual(len(plan.semantic_queries), 2)
        self.assertIn("chưa", plan.semantic_queries[1].text)

    def test_multi_issue_keeps_at_most_two_llm_subqueries(self) -> None:
        payload = base_payload(query_type="multi_issue", intent="procedure")
        payload["analysis"]["entities"] = [
            {"value": "giảm vốn", "evidence": "giảm vốn"}
        ]
        payload["analysis"]["must_terms"] = [
            {"value": "giảm vốn", "evidence": "giảm vốn"}
        ]
        payload["subqueries"] = [
            {"text": "Thủ tục đăng ký giảm vốn", "reason": "procedure"},
            {
                "text": "Thời hạn thực hiện thủ tục đăng ký giảm vốn",
                "reason": "deadline",
            },
            {"text": "Subquery thừa", "reason": "overflow"},
        ]
        planner, _ = planner_with_payload(payload)

        plan = planner.execute("Thủ tục đăng ký giảm vốn và thời hạn thực hiện là bao lâu?")

        self.assertEqual(len(plan.subqueries), 2)
        self.assertTrue(all("giảm vốn" in item.text for item in plan.subqueries))

    def test_fake_evidence_is_rejected(self) -> None:
        payload = base_payload()
        payload["analysis"]["entities"] = [
            {"value": "vốn điều lệ", "evidence": "thuế thu nhập doanh nghiệp"}
        ]
        planner, _ = planner_with_payload(payload)

        with self.assertRaisesRegex(ValueError, "evidence"):
            planner.execute("Thời hạn góp vốn là bao lâu?")

    def test_policy_rejects_numbers_invented_by_expansion(self) -> None:
        query = LegalQuery("Thời hạn góp vốn là bao lâu?")
        analysis = QueryAnalysis(
            intent="deadline",
            query_type=QueryType.LEGAL_CONCEPT,
            legal_domain="enterprise",
            entities=("góp vốn",),
            company_type=None,
            article_number=None,
            clause_number=None,
            document_number=None,
            temporal_scope=TemporalScope.CURRENT,
            must_terms=("góp vốn",),
        )
        proposal = ExpansionProposal(
            semantic_variants=(
                WeightedQuery(
                    "Thời hạn góp vốn là 90 ngày",
                    0.8,
                    QueryVariantKind.SEMANTIC,
                    "unsafe_claim",
                ),
            )
        )

        plan = QueryExpansionPolicy().build_plan(
            original_query=query,
            normalized_query=query,
            analysis=analysis,
            proposal=proposal,
        )

        self.assertEqual(len(plan.semantic_queries), 1)


if __name__ == "__main__":
    unittest.main()
