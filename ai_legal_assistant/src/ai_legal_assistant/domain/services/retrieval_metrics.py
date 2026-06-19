from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CaseRetrievalMetrics:
    recall_at_k: dict[int, float]
    reciprocal_rank_at_k: dict[int, float]


class RetrievalMetricsCalculator:
    def calculate(
        self,
        retrieved_ids: list[str],
        relevant_ids: frozenset[str],
        cutoffs: tuple[int, ...],
    ) -> CaseRetrievalMetrics:
        if not relevant_ids:
            raise ValueError("relevant_ids cannot be empty.")
        if not cutoffs or any(cutoff <= 0 for cutoff in cutoffs):
            raise ValueError("cutoffs must contain positive integers.")

        recall_at_k: dict[int, float] = {}
        reciprocal_rank_at_k: dict[int, float] = {}
        for cutoff in cutoffs:
            candidates = retrieved_ids[:cutoff]
            recall_at_k[cutoff] = len(set(candidates) & relevant_ids) / len(relevant_ids)
            reciprocal_rank_at_k[cutoff] = self._reciprocal_rank(candidates, relevant_ids)

        return CaseRetrievalMetrics(
            recall_at_k=recall_at_k,
            reciprocal_rank_at_k=reciprocal_rank_at_k,
        )

    @staticmethod
    def _reciprocal_rank(candidates: list[str], relevant_ids: frozenset[str]) -> float:
        for rank, candidate_id in enumerate(candidates, start=1):
            if candidate_id in relevant_ids:
                return 1.0 / rank
        return 0.0
