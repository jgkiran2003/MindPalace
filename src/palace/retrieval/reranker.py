"""FlashRank implementation of the reranking protocol."""

from __future__ import annotations

from typing import Any

from palace.config import Settings, get_settings


class FlashRankReranker:
    def __init__(self, settings: Settings | None = None, *, ranker: Any | None = None) -> None:
        self.settings = settings or get_settings()
        if ranker is None:
            from rerankers import Reranker

            ranker = Reranker(
                model_name=self.settings.reranker.model,
                model_type="flashrank",
                verbose=0,
            )
        self.ranker = ranker

    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
        scores: list[float] = []
        for query, document in pairs:
            value = self.ranker.score(query, document)
            if hasattr(value, "item"):
                value = value.item()
            scores.append(float(value))
        return scores

