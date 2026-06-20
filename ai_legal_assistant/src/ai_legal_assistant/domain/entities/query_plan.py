from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ai_legal_assistant.domain.entities.legal_query import LegalQuery


class QueryType(str, Enum):
    EXACT_LOOKUP = "exact_lookup"
    LEGAL_CONCEPT = "legal_concept"
    LEGAL_SITUATION = "legal_situation"
    MULTI_ISSUE = "multi_issue"
    AMBIGUOUS = "ambiguous"


class TemporalScope(str, Enum):
    CURRENT = "current"
    HISTORICAL = "historical"


class QueryVariantKind(str, Enum):
    ORIGINAL = "original"
    SEMANTIC = "semantic"
    SCOPE = "scope"
    SUBQUERY = "subquery"


@dataclass(frozen=True)
class AnalysisEvidence:
    field: str
    value: str
    quote: str


@dataclass(frozen=True)
class QueryAnalysis:
    intent: str
    query_type: QueryType
    legal_domain: str | None
    entities: tuple[str, ...]
    company_type: str | None
    article_number: str | None
    clause_number: str | None
    document_number: str | None
    temporal_scope: TemporalScope
    must_terms: tuple[str, ...]
    evidence: tuple[AnalysisEvidence, ...] = ()


@dataclass(frozen=True)
class WeightedQuery:
    text: str
    weight: float
    kind: QueryVariantKind
    reason: str
    scope: str | None = None

    def __post_init__(self) -> None:
        normalized = self.text.strip()
        if not normalized:
            raise ValueError("Weighted query text cannot be empty.")
        if not 0 < self.weight <= 1:
            raise ValueError("Weighted query weight must be in the interval (0, 1].")
        object.__setattr__(self, "text", normalized)


@dataclass(frozen=True)
class ExpansionProposal:
    semantic_variants: tuple[WeightedQuery, ...] = ()
    subqueries: tuple[WeightedQuery, ...] = ()
    lexical_terms: tuple[str, ...] = ()


@dataclass(frozen=True)
class QueryPlanningDraft:
    analysis: QueryAnalysis
    expansion: ExpansionProposal


@dataclass(frozen=True)
class QueryPlan:
    original_query: LegalQuery
    normalized_query: LegalQuery
    analysis: QueryAnalysis
    semantic_queries: tuple[WeightedQuery, ...]
    subqueries: tuple[WeightedQuery, ...]
    lexical_terms: tuple[str, ...]
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.semantic_queries:
            raise ValueError("Query plan must contain the original query.")
        first = self.semantic_queries[0]
        if first.kind != QueryVariantKind.ORIGINAL or first.weight != 1.0:
            raise ValueError("The first semantic query must be the original query with weight 1.0.")
        if len(self.semantic_queries) > 4:
            raise ValueError("Query plan cannot contain more than three semantic variants.")
        if len(self.subqueries) > 2:
            raise ValueError("Query plan cannot contain more than two subqueries.")
