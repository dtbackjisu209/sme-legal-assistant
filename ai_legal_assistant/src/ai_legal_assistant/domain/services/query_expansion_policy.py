from __future__ import annotations

from dataclasses import dataclass
import re
from typing import ClassVar

from ai_legal_assistant.domain.entities.legal_query import LegalQuery
from ai_legal_assistant.domain.entities.query_plan import (
    ExpansionProposal,
    QueryAnalysis,
    QueryPlan,
    QueryType,
    QueryVariantKind,
    WeightedQuery,
)


@dataclass(frozen=True)
class QueryExpansionPolicy:
    COMPANY_TYPE_MARKERS: ClassVar[tuple[tuple[str, str], ...]] = (
        ("trách nhiệm hữu hạn một thành viên", "limited_liability_one_member"),
        ("trách nhiệm hữu hạn hai thành viên", "limited_liability_two_or_more"),
        ("trách nhiệm hữu hạn từ hai thành viên", "limited_liability_two_or_more"),
        ("công ty cổ phần", "joint_stock"),
        ("công ty hợp danh", "partnership"),
        ("doanh nghiệp tư nhân", "private_enterprise"),
        ("hợp tác xã", "cooperative"),
        ("trách nhiệm hữu hạn", "limited_liability"),
    )
    max_semantic_variants: int = 3
    max_subqueries: int = 2

    def build_original_only_plan(
        self,
        *,
        original_query: LegalQuery,
        normalized_query: LegalQuery,
        analysis: QueryAnalysis,
        warning: str,
    ) -> QueryPlan:
        original = WeightedQuery(
            text=normalized_query.text,
            weight=1.0,
            kind=QueryVariantKind.ORIGINAL,
            reason="original_user_query",
        )
        return QueryPlan(
            original_query=original_query,
            normalized_query=normalized_query,
            analysis=analysis,
            semantic_queries=(original,),
            subqueries=(),
            lexical_terms=(),
            warnings=(warning,),
        )

    def build_plan(
        self,
        *,
        original_query: LegalQuery,
        normalized_query: LegalQuery,
        analysis: QueryAnalysis,
        proposal: ExpansionProposal,
    ) -> QueryPlan:
        original = WeightedQuery(
            text=normalized_query.text,
            weight=1.0,
            kind=QueryVariantKind.ORIGINAL,
            reason="original_user_query",
        )
        if analysis.query_type == QueryType.EXACT_LOOKUP:
            semantic_variants: tuple[WeightedQuery, ...] = ()
            subqueries: tuple[WeightedQuery, ...] = ()
        else:
            semantic_variants = self._select_unique(
                proposal.semantic_variants,
                normalized_query.text,
                analysis,
                limit=self.max_semantic_variants,
            )
            subqueries = self._select_unique(
                proposal.subqueries,
                normalized_query.text,
                analysis,
                limit=self.max_subqueries,
            )
        if analysis.query_type == QueryType.AMBIGUOUS and len(semantic_variants) < 2:
            raise ValueError(
                "Ambiguous query has fewer than two safe expansion variants after validation."
            )
        if analysis.query_type == QueryType.MULTI_ISSUE and not subqueries:
            raise ValueError("Multi-issue query has no safe subqueries after validation.")

        return QueryPlan(
            original_query=original_query,
            normalized_query=normalized_query,
            analysis=analysis,
            semantic_queries=(original, *semantic_variants),
            subqueries=subqueries,
            lexical_terms=self._safe_lexical_terms(
                proposal.lexical_terms,
                normalized_query.text,
                (*semantic_variants, *subqueries),
            ),
        )

    @staticmethod
    def _select_unique(
        candidates: tuple[WeightedQuery, ...],
        original_text: str,
        analysis: QueryAnalysis,
        *,
        limit: int,
    ) -> tuple[WeightedQuery, ...]:
        selected: list[WeightedQuery] = []
        seen = {original_text.casefold().strip()}
        for candidate in candidates:
            if not QueryExpansionPolicy._is_safe_candidate(
                candidate.text,
                original_text,
                analysis,
            ):
                continue
            key = candidate.text.casefold().strip()
            if key in seen:
                continue
            seen.add(key)
            selected.append(candidate)
            if len(selected) >= limit:
                break
        return tuple(selected)

    @staticmethod
    def _is_safe_candidate(
        candidate: str,
        original: str,
        analysis: QueryAnalysis,
    ) -> bool:
        original_folded = original.casefold()
        candidate_folded = candidate.casefold()
        for marker in ("không được", "không phải", "không", "chưa"):
            if marker in original_folded and marker not in candidate_folded:
                return False

        original_numbers = set(re.findall(r"\b\d+(?:[./-]\w+)*", original_folded))
        candidate_numbers = set(re.findall(r"\b\d+(?:[./-]\w+)*", candidate_folded))
        if candidate_numbers - original_numbers:
            return False

        if analysis.article_number is None and re.search(r"\bđiều\s+\d", candidate_folded):
            return False
        if not QueryExpansionPolicy._preserves_company_type(candidate_folded, analysis):
            return False
        return True

    @staticmethod
    def _preserves_company_type(candidate: str, analysis: QueryAnalysis) -> bool:
        expected = analysis.company_type
        if expected is None:
            return True
        detected = next(
            (
                company_type
                for marker, company_type in QueryExpansionPolicy.COMPANY_TYPE_MARKERS
                if marker in candidate
            ),
            None,
        )
        if expected == "limited_liability":
            return detected in {
                "limited_liability",
                "limited_liability_one_member",
                "limited_liability_two_or_more",
            }
        return detected == expected

    @staticmethod
    def _safe_lexical_terms(
        terms: tuple[str, ...],
        original_text: str,
        accepted_queries: tuple[WeightedQuery, ...],
    ) -> tuple[str, ...]:
        result: list[str] = []
        seen: set[str] = set()
        original_numbers = set(re.findall(r"\b\d+(?:[./-]\w+)*", original_text.casefold()))
        grounded_text = " ".join(
            (original_text, *(query.text for query in accepted_queries))
        ).casefold()
        for term in terms:
            value = term.strip()
            key = value.casefold()
            term_numbers = set(re.findall(r"\b\d+(?:[./-]\w+)*", key))
            if (
                value
                and key not in seen
                and not term_numbers - original_numbers
                and not re.search(r"\bđiều\s+\d", key)
                and key in grounded_text
            ):
                seen.add(key)
                result.append(value)
            if len(result) >= 12:
                break
        return tuple(result)
