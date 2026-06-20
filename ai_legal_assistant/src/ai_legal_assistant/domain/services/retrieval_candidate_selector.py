from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from ai_legal_assistant.domain.entities.retrieval_candidate import RerankedCandidate


@dataclass(frozen=True)
class RetrievalCandidateSelector:
    def select(
        self,
        candidates: tuple[RerankedCandidate, ...],
        *,
        top_k: int,
        required_scopes: tuple[str, ...] = (),
        max_chunks_per_article: int = 2,
        restrict_to_required_scopes: bool = False,
        scope_relevance_margin: float | None = None,
    ) -> tuple[RerankedCandidate, ...]:
        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        if max_chunks_per_article <= 0:
            raise ValueError("max_chunks_per_article must be positive.")
        if scope_relevance_margin is not None and scope_relevance_margin < 0:
            raise ValueError("scope_relevance_margin cannot be negative.")

        ranked = sorted(
            candidates,
            key=lambda item: (item.relevance_score, item.candidate.rrf_score),
            reverse=True,
        )
        if restrict_to_required_scopes:
            required_scope_set = set(required_scopes)
            ranked = [
                item
                for item in ranked
                if required_scope_set.intersection(item.candidate.matched_scopes)
            ]
        selected: list[RerankedCandidate] = []
        selected_ids: set[str] = set()
        article_counts: Counter[str] = Counter()

        for scope in dict.fromkeys(required_scopes):
            match = next(
                (
                    item
                    for item in ranked
                    if scope in item.candidate.matched_scopes
                    and self._can_select(
                        item,
                        selected_ids=selected_ids,
                        article_counts=article_counts,
                        max_chunks_per_article=max_chunks_per_article,
                    )
                ),
                None,
            )
            if match is not None:
                self._append(match, selected, selected_ids, article_counts)

        selected.sort(
            key=lambda item: (item.relevance_score, item.candidate.rrf_score),
            reverse=True,
        )
        relevance_floor = self._relevance_floor(selected, scope_relevance_margin)
        for item in ranked:
            if len(selected) >= top_k:
                break
            if relevance_floor is not None and item.relevance_score < relevance_floor:
                continue
            if self._can_select(
                item,
                selected_ids=selected_ids,
                article_counts=article_counts,
                max_chunks_per_article=max_chunks_per_article,
            ):
                self._append(item, selected, selected_ids, article_counts)
        return tuple(selected[:top_k])

    @staticmethod
    def _relevance_floor(
        selected_scope_winners: list[RerankedCandidate],
        scope_relevance_margin: float | None,
    ) -> float | None:
        if not selected_scope_winners or scope_relevance_margin is None:
            return None
        return min(item.relevance_score for item in selected_scope_winners) - scope_relevance_margin

    @staticmethod
    def _can_select(
        item: RerankedCandidate,
        *,
        selected_ids: set[str],
        article_counts: Counter[str],
        max_chunks_per_article: int,
    ) -> bool:
        content_id = item.candidate.content_id
        article_id = str(item.candidate.metadata.get("article_id") or "")
        return content_id not in selected_ids and (
            not article_id or article_counts[article_id] < max_chunks_per_article
        )

    @staticmethod
    def _append(
        item: RerankedCandidate,
        selected: list[RerankedCandidate],
        selected_ids: set[str],
        article_counts: Counter[str],
    ) -> None:
        selected.append(item)
        selected_ids.add(item.candidate.content_id)
        article_id = str(item.candidate.metadata.get("article_id") or "")
        if article_id:
            article_counts[article_id] += 1
