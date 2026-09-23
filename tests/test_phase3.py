from __future__ import annotations

import threading

import pytest

from palace.config import Settings, get_settings
from palace.domain.models import (
    Claim,
    ClaimProposal,
    ClaimStatus,
    ChunkStatus,
    ConflictGroup,
    DocumentChunk,
    SourceType,
)
from palace.retrieval.hybrid_searcher import HybridSearcher, fuse_rrf


def _vector(position: int = 0) -> list[float]:
    vector = [0.0] * 768
    vector[position] = 1.0
    return vector


def _chunk(
    chunk_id: str,
    text: str,
    *,
    status: ChunkStatus = ChunkStatus.COMPILED,
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        doc_path=f"{chunk_id}.md",
        chunk_index=0,
        text=text,
        vector=_vector(),
        status=status,
        source_type=SourceType.DOCUMENT,
    )


def _claim(
    claim_id: str,
    value: str,
    *,
    status: ClaimStatus = ClaimStatus.ACTIVE,
) -> Claim:
    return Claim(
        id=claim_id,
        entity="user",
        attribute="employer",
        value=value,
        status=status,
        source_type=SourceType.CHAT,
    )


def test_three_way_rrf_math_and_candidate_deduplication() -> None:
    shared_chunk = _chunk("shared", "shared chunk")
    other_chunk = _chunk("other", "other chunk")
    same_id_claim = _claim("shared", "claim with a colliding ID")

    candidates = fuse_rrf(
        [shared_chunk, other_chunk, shared_chunk],
        [other_chunk, shared_chunk],
        [same_id_claim, same_id_claim],
        rrf_k=60,
        limit=40,
    )

    by_key = {(candidate.kind, candidate.id): candidate for candidate in candidates}
    assert len(candidates) == 3
    assert by_key[("chunk", "shared")].rrf_score == pytest.approx(
        1.0 / 61 + 1.0 / 62
    )
    assert by_key[("chunk", "other")].rrf_score == pytest.approx(
        1.0 / 62 + 1.0 / 61
    )
    assert by_key[("claim", "shared")].rrf_score == pytest.approx(1.0 / 61)


class _Embedder:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def embed(self, query: str) -> list[float]:
        self.queries.append(query)
        return _vector()


class _FileStore:
    def __init__(
        self,
        barrier: threading.Barrier,
        raw: DocumentChunk,
        compiled: DocumentChunk,
    ) -> None:
        self.barrier = barrier
        self.raw = raw
        self.compiled = compiled
        self.vector_limit: int | None = None
        self.fts_limit: int | None = None

    def vector_search(
        self,
        query_vector: list[float],
        limit: int,
    ) -> list[DocumentChunk]:
        assert query_vector == _vector()
        self.vector_limit = limit
        self.barrier.wait()
        return [self.raw, self.compiled]

    def fts_search(self, query_text: str, limit: int) -> list[DocumentChunk]:
        assert query_text == "where do I work"
        self.fts_limit = limit
        self.barrier.wait()
        return [self.raw]

    def upsert_chunks(self, chunks: list[DocumentChunk]) -> None:
        raise NotImplementedError

    def delete_chunks_by_doc_path(self, doc_path: str) -> int:
        raise NotImplementedError

    def optimize(self) -> None:
        raise NotImplementedError


class _ClaimsStore:
    def __init__(
        self,
        barrier: threading.Barrier,
        claim: Claim,
        group: ConflictGroup,
    ) -> None:
        self.barrier = barrier
        self.claim = claim
        self.group = group
        self.search_k: int | None = None
        self.conflict_keys: list[str] = []

    def search_claims(
        self,
        query_embedding: list[float],
        query_text: str,
        k: int,
        status_filter: list[ClaimStatus] | None = None,
    ) -> list[Claim]:
        assert query_embedding == _vector()
        assert query_text == "where do I work"
        assert status_filter is None
        self.search_k = k
        self.barrier.wait()
        return [self.claim]

    def get_conflict_group(self, claim_key: str) -> ConflictGroup | None:
        self.conflict_keys.append(claim_key)
        return self.group if claim_key == self.group.claim_key else None

    def get_active_claim(self, entity: str, attribute: str) -> Claim | None:
        raise NotImplementedError

    def propose_claim(
        self,
        proposal: ClaimProposal,
        embedding: list[float],
    ) -> Claim:
        raise NotImplementedError


class _Reranker:
    def __init__(self) -> None:
        self.pairs: list[tuple[str, str]] = []

    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
        self.pairs = pairs
        return [1.0] * len(pairs)


@pytest.mark.asyncio
async def test_async_retrieve_penalties_and_conflict_bundling() -> None:
    raw = _chunk("z-raw", "raw note", status=ChunkStatus.RAW)
    compiled = _chunk("b-compiled", "compiled note")
    conflicted = _claim(
        "a-claim",
        "Freelance",
        status=ClaimStatus.NEEDS_REVIEW,
    )
    sibling = _claim(
        "sibling",
        "MyStorage",
        status=ClaimStatus.NEEDS_REVIEW,
    )
    group = ConflictGroup(
        claim_key=conflicted.claim_key,
        claims=[conflicted, sibling],
    )

    barrier = threading.Barrier(3, timeout=5)
    file_store = _FileStore(barrier, raw, compiled)
    claims_store = _ClaimsStore(barrier, conflicted, group)
    embedder = _Embedder()
    reranker = _Reranker()
    base = get_settings()
    settings: Settings = base.model_copy(
        update={
            "retrieval": base.retrieval.model_copy(
                update={
                    "top_k": 40,
                    "retrieval_top_k": 8,
                    "claims_search_k": 20,
                    "rrf_k": 60,
                    "raw_status_penalty": 0.4,
                }
            )
        }
    )
    searcher = HybridSearcher(
        file_store=file_store,
        claims_store=claims_store,
        embedder=embedder,  # type: ignore[arg-type]
        reranker=reranker,
        settings=settings,
    )

    results = await searcher.retrieve("where do I work")

    assert embedder.queries == ["where do I work"]
    assert file_store.vector_limit == settings.retrieval.top_k
    assert file_store.fts_limit == settings.retrieval.top_k
    assert claims_store.search_k == settings.retrieval.claims_search_k
    assert len(reranker.pairs) == 3

    by_id = {candidate.id: candidate for candidate in results}
    assert by_id[raw.chunk_id].flashrank_score == pytest.approx(1.0)
    assert by_id[raw.chunk_id].final_score == pytest.approx(0.4)
    assert by_id[compiled.chunk_id].final_score == pytest.approx(1.0)
    assert by_id[conflicted.id].final_score == pytest.approx(1.0)
    assert [candidate.id for candidate in results] == [
        conflicted.id,
        compiled.chunk_id,
        raw.chunk_id,
    ]

    claim_candidate = by_id[conflicted.id]
    assert claim_candidate.conflict_group is group
    assert claims_store.conflict_keys == [conflicted.claim_key]
    assert {claim.value for claim in claim_candidate.conflict_group.claims} == {
        "Freelance",
        "MyStorage",
    }
