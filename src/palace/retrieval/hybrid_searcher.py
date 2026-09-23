"""Three-leg hybrid retrieval orchestration for MindPalace."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

from palace.config import Settings, get_settings
from palace.domain.models import Claim, ChunkStatus, DocumentChunk
from palace.interfaces import ClaimsStoreAdapter, FileStore, FlashRankReranker
from palace.llm.embedder import OllamaEmbedder
from palace.retrieval.conflict_bundler import bundle_conflicts
from palace.retrieval.retrieval_types import Candidate


def _chunk_candidate(chunk: DocumentChunk) -> Candidate:
    return Candidate(
        id=chunk.chunk_id,
        text=chunk.text,
        kind="chunk",
        status=chunk.status.value,
        payload=chunk,
    )


def _claim_candidate(claim: Claim) -> Candidate:
    return Candidate(
        id=claim.id,
        text=f"{claim.claim_key}: {claim.value}",
        kind="claim",
        status=claim.status.value,
        payload=claim,
    )


def fuse_rrf(
    vector_results: Sequence[DocumentChunk],
    fts_results: Sequence[DocumentChunk],
    claim_results: Sequence[Claim],
    *,
    rrf_k: int,
    limit: int,
) -> list[Candidate]:
    """Fuse three ranked legs with 1-indexed reciprocal-rank fusion."""
    candidates: dict[tuple[str, str], Candidate] = {}
    legs: tuple[tuple[str, Sequence[DocumentChunk | Claim]], ...] = (
        ("chunk", vector_results),
        ("chunk", fts_results),
        ("claim", claim_results),
    )

    for kind, results in legs:
        seen_in_leg: set[tuple[str, str]] = set()
        for rank, payload in enumerate(results, start=1):
            if kind == "chunk":
                assert isinstance(payload, DocumentChunk)
                candidate = _chunk_candidate(payload)
            else:
                assert isinstance(payload, Claim)
                candidate = _claim_candidate(payload)
            key = (candidate.kind, candidate.id)
            if key in seen_in_leg:
                continue
            seen_in_leg.add(key)
            stored = candidates.setdefault(key, candidate)
            stored.rrf_score += 1.0 / (rrf_k + rank)

    return sorted(
        candidates.values(),
        key=lambda candidate: (
            -candidate.rrf_score,
            candidate.id,
            candidate.kind,
        ),
    )[:limit]


def rerank_and_penalize(
    query: str,
    candidates: list[Candidate],
    reranker: FlashRankReranker,
    *,
    raw_status_penalty: float,
    limit: int,
) -> list[Candidate]:
    """Apply FlashRank scores first and raw-chunk weighting second."""
    scores = reranker.score([(query, candidate.text) for candidate in candidates])
    if len(scores) != len(candidates):
        raise ValueError("reranker returned a different number of scores than candidates")

    for candidate, score in zip(candidates, scores, strict=True):
        candidate.flashrank_score = float(score)
        status_weight = (
            raw_status_penalty
            if candidate.kind == "chunk"
            and candidate.status == ChunkStatus.RAW.value
            else 1.0
        )
        candidate.final_score = candidate.flashrank_score * status_weight

    return sorted(
        candidates,
        key=lambda candidate: (-candidate.final_score, candidate.id),
    )[:limit]


class HybridSearcher:
    """Run vector, full-text, and claims retrieval as one ranked pipeline."""

    def __init__(
        self,
        file_store: FileStore,
        claims_store: ClaimsStoreAdapter,
        embedder: OllamaEmbedder,
        reranker: FlashRankReranker,
        settings: Settings | None = None,
    ) -> None:
        self.file_store = file_store
        self.claims_store = claims_store
        self.embedder = embedder
        self.reranker = reranker
        self.settings = settings or get_settings()

    async def retrieve(self, query: str) -> list[Candidate]:
        query_vector = await asyncio.to_thread(self.embedder.embed, query)
        retrieval = self.settings.retrieval

        vector_results, fts_results, claim_results = await asyncio.gather(
            asyncio.to_thread(
                self.file_store.vector_search,
                query_vector,
                limit=retrieval.top_k,
            ),
            asyncio.to_thread(
                self.file_store.fts_search,
                query,
                limit=retrieval.top_k,
            ),
            asyncio.to_thread(
                self.claims_store.search_claims,
                query_vector,
                query_text=query,
                k=retrieval.claims_search_k,
            ),
        )

        candidates = fuse_rrf(
            vector_results,
            fts_results,
            claim_results,
            rrf_k=retrieval.rrf_k,
            limit=retrieval.top_k,
        )
        final_results = await asyncio.to_thread(
            rerank_and_penalize,
            query,
            candidates,
            self.reranker,
            raw_status_penalty=retrieval.raw_status_penalty,
            limit=retrieval.retrieval_top_k,
        )
        return await asyncio.to_thread(
            bundle_conflicts,
            final_results,
            self.claims_store,
        )
