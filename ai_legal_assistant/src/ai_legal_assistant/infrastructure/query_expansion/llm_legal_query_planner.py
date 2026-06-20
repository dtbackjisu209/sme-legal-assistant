from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from ai_legal_assistant.application.ports.llm_port import TextGenerationPort
from ai_legal_assistant.application.ports.query_planning_port import (
    QueryExpansionRejectedError,
)
from ai_legal_assistant.domain.entities.legal_query import LegalQuery
from ai_legal_assistant.domain.entities.query_plan import (
    AnalysisEvidence,
    ExpansionProposal,
    QueryAnalysis,
    QueryPlanningDraft,
    QueryType,
    QueryVariantKind,
    TemporalScope,
    WeightedQuery,
)


SYSTEM_PROMPT = """Bạn là bộ lập kế hoạch truy xuất pháp luật Việt Nam.
Chỉ phân tích QUERY được cung cấp. Không trả lời câu hỏi pháp luật và không dùng kiến thức
bên ngoài để thêm điều luật, thời hạn, mức phạt, số tiền hoặc sự kiện.

Yêu cầu bắt buộc:
1. Giữ nguyên chủ thể, phủ định, mốc thời gian và loại hình tổ chức trong QUERY.
2. Chỉ đặt company_type/article_number/clause_number/document_number khi QUERY nói rõ.
   Mỗi giá trị trích xuất phải có evidence là đoạn trích nguyên văn từ QUERY.
3. exact_lookup: semantic_variants và subqueries phải rỗng.
4. legal_concept/legal_situation: được tạo tối đa 3 cách diễn đạt cùng nghĩa.
5. multi_issue: tối đa 2 subqueries, mỗi subquery phải giữ đủ ngữ cảnh chủ thể.
6. ambiguous: chỉ phân nhánh theo đối tượng hợp lý, không khẳng định đáp án.
7. Variant không được chứa con số hoặc viện dẫn mới không có trong QUERY.
8. Không dùng HyDE. Không tạo đoạn văn giống câu trả lời.
9. Chỉ xuất đúng một JSON object, không markdown, không giải thích.

Quy tắc phân loại quan trọng:
- exact_lookup CHỈ khi QUERY nêu rõ Điều, Khoản, Điểm hoặc số hiệu văn bản.
- legal_situation khi QUERY mô tả sự việc thực tế của người hỏi.
- multi_issue khi QUERY có từ hai vấn đề độc lập cần truy xuất riêng.
- ambiguous khi thiếu thông tin làm căn cứ pháp lý thay đổi theo đối tượng/loại hình.
- legal_concept cho câu hỏi khái niệm hoặc quy định chung còn lại.
- entity và must_term phải là cụm pháp lý ngắn, không lấy "bao lâu", "thế nào".

Giá trị hợp lệ:
- intent: deadline, penalty, procedure, definition, obligation, eligibility, unknown
- query_type: exact_lookup, legal_concept, legal_situation, multi_issue, ambiguous
- temporal_scope: current, historical
- company_type: limited_liability, limited_liability_one_member,
  limited_liability_two_or_more, joint_stock, partnership, private_enterprise,
  cooperative hoặc null
- variant kind: semantic hoặc scope

JSON schema logic:
{
  "analysis": {
    "intent": "...",
    "query_type": "...",
    "legal_domain": "enterprise|labor|tax|social_insurance|civil|investment|securities|unknown",
    "entities": [{"value": "...", "evidence": "trích nguyên văn"}],
    "company_type": {"value": "...", "evidence": "trích nguyên văn"} hoặc null,
    "article_number": {"value": "...", "evidence": "trích nguyên văn"} hoặc null,
    "clause_number": {"value": "...", "evidence": "trích nguyên văn"} hoặc null,
    "document_number": {"value": "...", "evidence": "trích nguyên văn"} hoặc null,
    "temporal_scope": "current|historical",
    "must_terms": [{"value": "...", "evidence": "trích nguyên văn"}]
  },
  "semantic_variants": [{"text": "...", "kind": "semantic", "reason": "..."}],
  "subqueries": [{"text": "...", "reason": "..."}],
  "lexical_terms": ["..."]
}

Ví dụ bắt buộc phải học theo:
QUERY: Thời hạn góp đủ vốn điều lệ là bao lâu?
OUTPUT:
{"analysis":{"intent":"deadline","query_type":"ambiguous","legal_domain":"enterprise","entities":[{"value":"vốn điều lệ","evidence":"vốn điều lệ"}],"company_type":null,"article_number":null,"clause_number":null,"document_number":null,"temporal_scope":"current","must_terms":[{"value":"vốn điều lệ","evidence":"vốn điều lệ"}]},"semantic_variants":[{"text":"Thời hạn chủ sở hữu công ty trách nhiệm hữu hạn một thành viên góp đủ vốn điều lệ","kind":"scope","reason":"phân nhánh loại hình"},{"text":"Thời hạn thành viên công ty trách nhiệm hữu hạn từ hai thành viên góp đủ phần vốn cam kết","kind":"scope","reason":"phân nhánh loại hình"},{"text":"Thời hạn cổ đông công ty cổ phần thanh toán đủ cổ phần đăng ký mua","kind":"scope","reason":"phân nhánh loại hình"}],"subqueries":[],"lexical_terms":["góp vốn","vốn điều lệ","vốn cam kết"]}

QUERY: Khoản 2 Điều 47 quy định gì về góp vốn?
OUTPUT phải có query_type="exact_lookup", article_number="47", clause_number="2",
semantic_variants=[] và subqueries=[].
"""


ANALYSIS_PROMPT = """Bạn chỉ phân tích query để lập kế hoạch truy xuất pháp luật Việt Nam.
Không trả lời câu hỏi, không sinh query mở rộng và không dùng kiến thức bên ngoài.
Chỉ trích xuất điều query nói rõ. Mỗi entity, must_term và constraint phải có evidence là
đoạn trích nguyên văn trong query.

Phân loại:
- exact_lookup CHỈ khi query nêu rõ Điều, Khoản, Điểm hoặc số hiệu văn bản.
- legal_situation khi query mô tả tình huống thực tế của người hỏi.
- multi_issue khi có từ hai vấn đề độc lập cần truy xuất riêng.
- ambiguous khi thiếu đối tượng/loại hình làm căn cứ pháp lý thay đổi.
- legal_concept cho câu hỏi pháp lý chung còn lại.

entity và must_term phải là cụm pháp lý ngắn; không dùng "bao lâu", "thế nào", "khi nào".
Chỉ xuất JSON, không markdown hay giải thích:
{"analysis":{"intent":"deadline|penalty|procedure|definition|obligation|eligibility|unknown","query_type":"exact_lookup|legal_concept|legal_situation|multi_issue|ambiguous","legal_domain":"enterprise|labor|tax|social_insurance|civil|investment|securities|unknown","entities":[{"value":"cụm pháp lý ngắn","evidence":"trích nguyên văn"}],"company_type":null,"article_number":null,"clause_number":null,"document_number":null,"temporal_scope":"current|historical","must_terms":[{"value":"cụm bắt buộc","evidence":"trích nguyên văn"}]},"semantic_variants":[],"subqueries":[],"lexical_terms":[]}

Ví dụ:
QUERY: Thời hạn góp đủ vốn điều lệ là bao lâu?
OUTPUT:
{"analysis":{"intent":"deadline","query_type":"ambiguous","legal_domain":"enterprise","entities":[{"value":"vốn điều lệ","evidence":"vốn điều lệ"}],"company_type":null,"article_number":null,"clause_number":null,"document_number":null,"temporal_scope":"current","must_terms":[{"value":"vốn điều lệ","evidence":"vốn điều lệ"}]},"semantic_variants":[],"subqueries":[],"lexical_terms":[]}

QUERY: Khoản 2 Điều 47 quy định gì về góp vốn?
OUTPUT phải có query_type="exact_lookup", article_number={"value":"47","evidence":"Điều 47"},
clause_number={"value":"2","evidence":"Khoản 2"} và các array đều rỗng.
"""


EXPANSION_REPAIR_PROMPT = """NHIỆM VỤ DUY NHẤT: tạo query expansion để truy xuất pháp luật.
Đây không phải câu hỏi cần trả lời. Không viết đáp án pháp luật.

RÀNG BUỘC CỨNG:
1. Không sao chép nguyên văn QUERY thành variant.
2. Mọi variant phải khác nhau về nhánh đối tượng, không được trùng nhau.
3. Không thêm số, thời hạn, mức phạt, số tiền, số điều hoặc số khoản không có trong QUERY.
4. Không viết kết luận dạng "là 30 ngày", "là 90 ngày" hoặc bất kỳ đáp án nào.
5. Giữ nguyên phủ định, chủ thể, thời gian và loại hình đã được nêu trong QUERY.
6. Chỉ xuất đúng một JSON object; không markdown, suy luận hoặc giải thích bên ngoài JSON.

Quy tắc:
- ambiguous: semantic_variants phải có ĐÚNG 3 phần tử; cả ba có kind="scope".
- legal_concept/legal_situation: tạo 1-3 semantic variants cùng nghĩa.
- multi_issue: tạo 1-2 subqueries, mỗi câu giữ đủ chủ thể và ngữ cảnh.
- exact_lookup: không tạo expansion.
- Không dùng HyDE.

Với query mơ hồ về thời hạn góp vốn điều lệ, các nhánh hợp lý gồm công ty trách nhiệm hữu
hạn một thành viên, công ty trách nhiệm hữu hạn từ hai thành viên và công ty cổ phần.
Ba phần tử bắt buộc theo đúng thứ tự:
- semantic_variants[0] chứa "công ty trách nhiệm hữu hạn một thành viên".
- semantic_variants[1] chứa "công ty trách nhiệm hữu hạn từ hai thành viên".
- semantic_variants[2] chứa "công ty cổ phần".
Output mẫu đầy đủ cho trường hợp này:
{"semantic_variants":[{"text":"Thời hạn chủ sở hữu công ty trách nhiệm hữu hạn một thành viên góp đủ vốn điều lệ","kind":"scope","reason":"nhánh một thành viên"},{"text":"Thời hạn thành viên công ty trách nhiệm hữu hạn từ hai thành viên góp đủ phần vốn cam kết","kind":"scope","reason":"nhánh từ hai thành viên"},{"text":"Thời hạn cổ đông công ty cổ phần thanh toán đủ cổ phần đăng ký mua","kind":"scope","reason":"nhánh công ty cổ phần"}],"subqueries":[],"lexical_terms":["góp vốn","vốn điều lệ","vốn cam kết"]}

Với loại khác, vẫn dùng đúng các key semantic_variants, subqueries, lexical_terms và chỉ
điền số phần tử phù hợp quy tắc ở trên.

Trước khi xuất JSON, tự kiểm tra thầm: đủ số phần tử; không trùng QUERY; không trùng nhau;
không có số mới; không phải câu trả lời; kind đúng. Không in checklist này.
"""


_EVIDENCED_VALUE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["value", "evidence"],
    "properties": {
        "value": {"type": "string"},
        "evidence": {"type": "string"},
    },
}

_ANALYSIS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "intent",
        "query_type",
        "legal_domain",
        "entities",
        "company_type",
        "article_number",
        "clause_number",
        "document_number",
        "temporal_scope",
        "must_terms",
    ],
    "properties": {
        "intent": {
            "enum": [
                "deadline",
                "penalty",
                "procedure",
                "definition",
                "obligation",
                "eligibility",
                "unknown",
            ]
        },
        "query_type": {
            "enum": [
                "exact_lookup",
                "legal_concept",
                "legal_situation",
                "multi_issue",
                "ambiguous",
            ]
        },
        "legal_domain": {
            "anyOf": [
                {
                    "enum": [
                        "enterprise",
                        "labor",
                        "tax",
                        "social_insurance",
                        "civil",
                        "investment",
                        "securities",
                        "unknown",
                    ]
                },
                {"type": "null"},
            ]
        },
        "entities": {"type": "array", "items": _EVIDENCED_VALUE_SCHEMA},
        "company_type": {
            "anyOf": [
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["value", "evidence"],
                    "properties": {
                        "value": {
                            "enum": [
                                "limited_liability",
                                "limited_liability_one_member",
                                "limited_liability_two_or_more",
                                "joint_stock",
                                "partnership",
                                "private_enterprise",
                                "cooperative",
                            ]
                        },
                        "evidence": {"type": "string"},
                    },
                },
                {"type": "null"},
            ]
        },
        "article_number": {"anyOf": [_EVIDENCED_VALUE_SCHEMA, {"type": "null"}]},
        "clause_number": {"anyOf": [_EVIDENCED_VALUE_SCHEMA, {"type": "null"}]},
        "document_number": {"anyOf": [_EVIDENCED_VALUE_SCHEMA, {"type": "null"}]},
        "temporal_scope": {"enum": ["current", "historical"]},
        "must_terms": {"type": "array", "items": _EVIDENCED_VALUE_SCHEMA},
    },
}

_EXPANSION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["semantic_variants", "subqueries", "lexical_terms"],
    "properties": {
        "semantic_variants": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["text", "kind", "reason"],
                "properties": {
                    "text": {"type": "string"},
                    "kind": {"enum": ["semantic", "scope"]},
                    "reason": {"type": "string"},
                },
            },
        },
        "subqueries": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["text", "reason"],
                "properties": {
                    "text": {"type": "string"},
                    "reason": {"type": "string"},
                },
            },
        },
        "lexical_terms": {"type": "array", "items": {"type": "string"}},
    },
}

_ANALYSIS_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["analysis", "semantic_variants", "subqueries", "lexical_terms"],
    "properties": {
        "analysis": _ANALYSIS_SCHEMA,
        **_EXPANSION_SCHEMA["properties"],
    },
}

ANALYSIS_REPAIR_PROMPT = """Repair the planner output for one Vietnamese legal query.
Return exactly one JSON object matching the supplied schema. Do not answer the legal question.
Preserve only facts and legal references explicitly present in the query. Do not use Markdown."""


@dataclass
class LLMLegalQueryPlanner:
    llm: TextGenerationPort

    ALLOWED_INTENTS = frozenset(
        {"deadline", "penalty", "procedure", "definition", "obligation", "eligibility", "unknown"}
    )
    ALLOWED_DOMAINS = frozenset(
        {"enterprise", "labor", "tax", "social_insurance", "civil", "investment", "securities"}
    )
    ALLOWED_COMPANY_TYPES = frozenset(
        {
            "limited_liability",
            "limited_liability_one_member",
            "limited_liability_two_or_more",
            "joint_stock",
            "partnership",
            "private_enterprise",
            "cooperative",
        }
    )

    def plan(self, query: LegalQuery) -> QueryPlanningDraft:
        raw_output = self.llm.generate(
            system_prompt=ANALYSIS_PROMPT,
            json_schema=_ANALYSIS_RESPONSE_SCHEMA,
            user_prompt=f"QUERY CHUẨN HÓA:\n{query.text}\n\nChỉ xuất JSON:",
        )
        analysis_draft = self._analysis_draft_with_repair(query, raw_output)
        if analysis_draft.analysis.query_type == QueryType.EXACT_LOOKUP:
            return QueryPlanningDraft(
                analysis=analysis_draft.analysis,
                expansion=ExpansionProposal(lexical_terms=analysis_draft.analysis.must_terms),
            )

        try:
            expanded_draft = self._repair_expansion(query, analysis_draft.analysis)
            self._validate_draft_consistency(expanded_draft, query)
        except ValueError as exc:
            try:
                expanded_draft = self._repair_expansion(
                    query,
                    analysis_draft.analysis,
                    rejection_reason=str(exc),
                )
                self._validate_draft_consistency(expanded_draft, query)
            except ValueError as repair_exc:
                raise QueryExpansionRejectedError(
                    "LLM expansion remained invalid after one repair; "
                    "retrieval fell back to the original query only.",
                    analysis_draft.analysis,
                ) from repair_exc
        return expanded_draft

    def _analysis_draft_with_repair(
        self,
        query: LegalQuery,
        raw_output: str,
    ) -> QueryPlanningDraft:
        try:
            payload = self._parse_json(raw_output)
            draft = self._to_draft(payload, query)
            self._validate_analysis_consistency(draft.analysis)
            return draft
        except ValueError as exc:
            repaired_output = self.llm.generate(
                system_prompt=ANALYSIS_REPAIR_PROMPT,
                user_prompt=(
                    f"QUERY:\n{query.text}\n\n"
                    f"The previous planner output was rejected: {exc}\n"
                    "Generate a corrected planner JSON object now."
                ),
                json_schema=_ANALYSIS_RESPONSE_SCHEMA,
            )
            try:
                payload = self._parse_json(repaired_output)
                draft = self._to_draft(payload, query)
                self._validate_analysis_consistency(draft.analysis)
                return draft
            except ValueError as repair_exc:
                raise ValueError(
                    "LLM analysis remained invalid after one repair."
                ) from repair_exc

    @staticmethod
    def _validate_analysis_consistency(analysis: QueryAnalysis) -> None:
        has_exact_reference = any(
            value is not None
            for value in (
                analysis.article_number,
                analysis.clause_number,
                analysis.document_number,
            )
        )
        if analysis.query_type == QueryType.EXACT_LOOKUP and not has_exact_reference:
            raise ValueError("LLM classified exact_lookup without an explicit legal reference.")

        forbidden_terms = ("bao lâu", "thế nào", "khi nào", "là gì")
        for value in (*analysis.entities, *analysis.must_terms):
            folded = value.casefold()
            if len(value.split()) > 10 or any(term in folded for term in forbidden_terms):
                raise ValueError("LLM analysis returned an ungrounded or overly broad term.")

    def _to_draft(self, payload: dict[str, Any], query: LegalQuery) -> QueryPlanningDraft:
        analysis_payload = self._mapping(payload.get("analysis"), "analysis")
        intent = self._allowed_string(
            analysis_payload.get("intent"),
            self.ALLOWED_INTENTS,
            "analysis.intent",
        )
        query_type = QueryType(self._string(analysis_payload.get("query_type"), "query_type"))
        temporal_scope = TemporalScope(
            self._string(analysis_payload.get("temporal_scope"), "temporal_scope")
        )
        legal_domain_value = analysis_payload.get("legal_domain")
        legal_domain = None
        if legal_domain_value not in (None, "unknown"):
            legal_domain = self._allowed_string(
                legal_domain_value,
                self.ALLOWED_DOMAINS,
                "analysis.legal_domain",
            )

        evidence: list[AnalysisEvidence] = []
        entities = self._evidenced_list(
            analysis_payload.get("entities"),
            field="entities",
            query=query.text,
            evidence=evidence,
        )
        must_terms = self._evidenced_list(
            analysis_payload.get("must_terms"),
            field="must_terms",
            query=query.text,
            evidence=evidence,
        )
        company_type = self._optional_evidenced_value(
            analysis_payload.get("company_type"),
            field="company_type",
            query=query.text,
            evidence=evidence,
            allowed=self.ALLOWED_COMPANY_TYPES,
        )
        article_number = self._optional_evidenced_value(
            analysis_payload.get("article_number"),
            field="article_number",
            query=query.text,
            evidence=evidence,
            require_value_in_evidence=True,
        )
        clause_number = self._optional_evidenced_value(
            analysis_payload.get("clause_number"),
            field="clause_number",
            query=query.text,
            evidence=evidence,
            require_value_in_evidence=True,
        )
        document_number = self._optional_evidenced_value(
            analysis_payload.get("document_number"),
            field="document_number",
            query=query.text,
            evidence=evidence,
            require_value_in_evidence=True,
        )

        analysis = QueryAnalysis(
            intent=intent,
            query_type=query_type,
            legal_domain=legal_domain,
            entities=entities,
            company_type=company_type,
            article_number=article_number,
            clause_number=clause_number,
            document_number=document_number,
            temporal_scope=temporal_scope,
            must_terms=must_terms,
            evidence=tuple(evidence),
        )
        return QueryPlanningDraft(
            analysis=analysis,
            expansion=self._expansion_from_payload(payload, analysis.query_type),
        )

    def _repair_expansion(
        self,
        query: LegalQuery,
        analysis: QueryAnalysis,
        rejection_reason: str | None = None,
    ) -> QueryPlanningDraft:
        analysis_context = {
            "intent": analysis.intent,
            "query_type": analysis.query_type.value,
            "legal_domain": analysis.legal_domain,
            "entities": list(analysis.entities),
            "company_type": analysis.company_type,
            "temporal_scope": analysis.temporal_scope.value,
            "must_terms": list(analysis.must_terms),
        }
        system_prompt = EXPANSION_REPAIR_PROMPT
        if rejection_reason is not None:
            system_prompt = (
                f"{EXPANSION_REPAIR_PROMPT}\n\n"
                "The previous expansion was rejected for this reason: "
                f"{rejection_reason}\n"
                "Generate a corrected expansion that satisfies every constraint."
            )
        raw_output = self.llm.generate(
            system_prompt=system_prompt,
            json_schema=_EXPANSION_SCHEMA,
            user_prompt=(
                f"QUERY CHUẨN HÓA:\n{query.text}\n\n"
                f"ANALYSIS ĐÃ XÁC THỰC:\n"
                f"{json.dumps(analysis_context, ensure_ascii=False)}\n\n"
                f"KHÔNG ĐƯỢC SAO CHÉP QUERY GỐC LÀM VARIANT:\n{query.text}\n\n"
                "Chỉ xuất JSON expansion:"
            ),
        )
        payload = self._parse_json(raw_output)
        return QueryPlanningDraft(
            analysis=analysis,
            expansion=self._expansion_from_payload(payload, analysis.query_type),
        )

    def _expansion_from_payload(
        self,
        payload: dict[str, Any],
        query_type: QueryType,
    ) -> ExpansionProposal:
        return ExpansionProposal(
            semantic_variants=self._variants(payload.get("semantic_variants"), query_type),
            subqueries=self._subqueries(payload.get("subqueries")),
            lexical_terms=self._string_list(payload.get("lexical_terms"), "lexical_terms"),
        )

    @staticmethod
    def _validate_draft_consistency(
        draft: QueryPlanningDraft,
        query: LegalQuery,
    ) -> None:
        analysis = draft.analysis
        expansion = draft.expansion
        has_exact_reference = any(
            value is not None
            for value in (
                analysis.article_number,
                analysis.clause_number,
                analysis.document_number,
            )
        )
        if analysis.query_type == QueryType.EXACT_LOOKUP and not has_exact_reference:
            raise ValueError("LLM classified exact_lookup without an explicit legal reference.")
        if analysis.query_type == QueryType.EXACT_LOOKUP and (
            expansion.semantic_variants or expansion.subqueries
        ):
            raise ValueError("LLM exact_lookup response must not contain expansion.")
        safe_semantic_variants = {
            " ".join(candidate.text.casefold().split()): candidate
            for candidate in expansion.semantic_variants
            if LLMLegalQueryPlanner._candidate_is_grounded(candidate, query)
        }
        safe_subqueries = tuple(
            candidate
            for candidate in expansion.subqueries
            if LLMLegalQueryPlanner._candidate_is_grounded(candidate, query)
        )
        if analysis.query_type == QueryType.AMBIGUOUS and len(safe_semantic_variants) < 2:
            raise ValueError("LLM ambiguous response must contain at least two scope variants.")
        if analysis.query_type == QueryType.MULTI_ISSUE and not safe_subqueries:
            raise ValueError("LLM multi_issue response must contain subqueries.")
        if analysis.query_type in {QueryType.LEGAL_CONCEPT, QueryType.LEGAL_SITUATION} and not (
            safe_semantic_variants or safe_subqueries
        ):
            raise ValueError("LLM response must contain at least one expansion candidate.")

    @staticmethod
    def _candidate_is_grounded(candidate: WeightedQuery, query: LegalQuery) -> bool:
        original_folded = query.text.casefold()
        original_numbers = set(re.findall(r"\b\d+(?:[./-]\w+)*", original_folded))
        candidate_folded = candidate.text.casefold()
        candidate_numbers = set(re.findall(r"\b\d+(?:[./-]\w+)*", candidate_folded))
        if candidate_numbers - original_numbers:
            return False
        for duration_unit in ("ngày", "tháng", "năm", "tuần", "giờ"):
            if duration_unit in candidate_folded and duration_unit not in original_folded:
                return False
        for marker in ("không được", "không phải", "không", "chưa"):
            if marker in original_folded and marker not in candidate_folded:
                return False
        return True

    def _variants(
        self,
        value: Any,
        query_type: QueryType,
    ) -> tuple[WeightedQuery, ...]:
        variants: list[WeightedQuery] = []
        for index, item in enumerate(self._list(value, "semantic_variants")):
            row = self._mapping(item, f"semantic_variants[{index}]")
            default_kind = (
                QueryVariantKind.SCOPE
                if query_type == QueryType.AMBIGUOUS
                else QueryVariantKind.SEMANTIC
            )
            kind_value = row.get("kind")
            kind = default_kind
            if kind_value in {QueryVariantKind.SEMANTIC.value, QueryVariantKind.SCOPE.value}:
                kind = QueryVariantKind(kind_value)
            variants.append(
                WeightedQuery(
                    text=self._string(row.get("text"), "variant.text"),
                    weight=0.8 if kind == QueryVariantKind.SEMANTIC else 0.7,
                    kind=kind,
                    reason=self._string(row.get("reason"), "variant.reason"),
                )
            )
        return tuple(variants)

    def _subqueries(self, value: Any) -> tuple[WeightedQuery, ...]:
        subqueries: list[WeightedQuery] = []
        for index, item in enumerate(self._list(value, "subqueries")):
            row = self._mapping(item, f"subqueries[{index}]")
            subqueries.append(
                WeightedQuery(
                    text=self._string(row.get("text"), "subquery.text"),
                    weight=0.75,
                    kind=QueryVariantKind.SUBQUERY,
                    reason=self._string(row.get("reason"), "subquery.reason"),
                )
            )
        return tuple(subqueries)

    def _evidenced_list(
        self,
        value: Any,
        *,
        field: str,
        query: str,
        evidence: list[AnalysisEvidence],
    ) -> tuple[str, ...]:
        result: list[str] = []
        for index, item in enumerate(self._list(value, field)):
            row = self._mapping(item, f"{field}[{index}]")
            extracted = self._string(row.get("value"), f"{field}.value")
            quote = self._string(row.get("evidence"), f"{field}.evidence")
            self._validate_quote(quote, query, field)
            result.append(extracted)
            evidence.append(AnalysisEvidence(field=field, value=extracted, quote=quote))
        return tuple(dict.fromkeys(result))

    def _optional_evidenced_value(
        self,
        value: Any,
        *,
        field: str,
        query: str,
        evidence: list[AnalysisEvidence],
        allowed: frozenset[str] | None = None,
        require_value_in_evidence: bool = False,
    ) -> str | None:
        if value is None:
            return None
        if isinstance(value, str) and value.strip().casefold() in {"null", "none"}:
            return None
        row = self._mapping(value, field)
        extracted = self._string(row.get("value"), f"{field}.value")
        if extracted.casefold() in {"null", "none"}:
            return None
        quote = self._string(row.get("evidence"), f"{field}.evidence")
        self._validate_quote(quote, query, field)
        if allowed is not None and extracted not in allowed:
            raise ValueError(f"Invalid {field}: {extracted}")
        if require_value_in_evidence and extracted.casefold() not in quote.casefold():
            raise ValueError(f"{field} value must occur in its evidence quote.")
        evidence.append(AnalysisEvidence(field=field, value=extracted, quote=quote))
        return extracted

    @staticmethod
    def _validate_quote(quote: str, query: str, field: str) -> None:
        normalized_quote = " ".join(quote.casefold().split())
        normalized_query = " ".join(query.casefold().split())
        if not normalized_quote or normalized_quote not in normalized_query:
            raise ValueError(f"LLM evidence for {field} is not present in the normalized query.")

    @staticmethod
    def _parse_json(raw_output: str) -> dict[str, Any]:
        text = raw_output.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\s*```$", "", text)
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end < start:
            raise ValueError("LLM query planner did not return a JSON object.")
        try:
            payload = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError("LLM query planner returned invalid JSON.") from exc
        if not isinstance(payload, dict):
            raise ValueError("LLM query planner response must be a JSON object.")
        return payload

    @staticmethod
    def _allowed_string(value: Any, allowed: frozenset[str], field: str) -> str:
        result = LLMLegalQueryPlanner._string(value, field)
        if result not in allowed:
            raise ValueError(f"Invalid {field}: {result}")
        return result

    @staticmethod
    def _string(value: Any, field: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} must be a non-empty string.")
        return value.strip()

    @staticmethod
    def _mapping(value: Any, field: str) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise ValueError(f"{field} must be a JSON object.")
        return value

    @staticmethod
    def _list(value: Any, field: str) -> list[Any]:
        if not isinstance(value, list):
            raise ValueError(f"{field} must be a JSON array.")
        return value

    @staticmethod
    def _string_list(value: Any, field: str) -> tuple[str, ...]:
        return tuple(
            LLMLegalQueryPlanner._string(item, f"{field}[]")
            for item in LLMLegalQueryPlanner._list(value, field)
        )
