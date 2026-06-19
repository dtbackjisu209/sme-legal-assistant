from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WeightedRankedList:
    item_ids: tuple[str, ...]
    weight: float


@dataclass(frozen=True)
class WeightedReciprocalRankFusion:
    rank_constant: int = 60

    def fuse(self, ranked_lists: tuple[WeightedRankedList, ...]) -> dict[str, float]:
        if self.rank_constant <= 0:
            raise ValueError("rank_constant must be positive.")

        scores: dict[str, float] = {}
        for ranked_list in ranked_lists:
            if not 0 < ranked_list.weight <= 1:
                raise ValueError("Ranked list weight must be in the interval (0, 1].")
            for rank, item_id in enumerate(ranked_list.item_ids, start=1):
                scores[item_id] = scores.get(item_id, 0.0) + (
                    ranked_list.weight / (self.rank_constant + rank)
                )
        return scores
