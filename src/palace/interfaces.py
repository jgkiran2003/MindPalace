"""
MindPalace — Component Interfaces
Structural (typing.Protocol) contracts for the four boundary components:
FileStore, ClaimsStoreAdapter, FlashRankReranker, HybridSearcher.

These are the locked module boundaries. Any concrete implementation
(e.g. `stores.file_store.LanceFileStore`) must satisfy the relevant
Protocol's signatures exactly — an implementer (human or Codex) should
treat this file as authoritative and not add, drop, or rename parameters.
"""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

from palace.domain.models import Claim, ClaimProposal, ClaimStatus, ConflictGroup, DocumentChunk
from palace.retrieval.retrieval_types import Candidate


@runtime_checkable
class FileStore(Protocol):
    """Adapter boundary over the LanceDB document-chunk table (lance_schema.py)."""

    def upsert_chunks(self, chunks: list[DocumentChunk]) -> None:
        """Insert or replace chunk rows. Idempotent on `chunk_id`."""
        ...

    def delete_chunks_by_doc_path(self, doc_path: str) -> int:
        """Remove every chunk belonging to a deleted/moved source file.
        Returns the number of chunks deleted."""
        ...

    def vector_search(self, query_vector: list[float], limit: int) -> list[DocumentChunk]:
        """ANN search over the `vector` column, nearest-first."""
        ...

    def fts_search(self, query_text: str, limit: int) -> list[DocumentChunk]:
        """Native Lance BM25 search over the `text` column."""
        ...

    def optimize(self) -> None:
        """Fold newly added rows into the vector and FTS indexes.
        Called by ingestion/indexer.py every settings.indexing.optimize_after_writes
        upserts; rows are not searchable via fts_search() until this runs."""
        ...


@runtime_checkable
class ClaimsStoreAdapter(Protocol):
    """Adapter boundary over the SQLite + sqlite-vec claims store (schema.sql)."""

    def get_active_claim(self, entity: str, attribute: str) -> Optional[Claim]:
        """Direct indexed lookup for tools that need one specific fact."""
        ...

    def search_claims(
        self,
        query_embedding: list[float],
        query_text: str,
        k: int,
        status_filter: Optional[list[ClaimStatus]] = None,
    ) -> list[Claim]:
        """Internally fused (vector + text) ranked claim list, already
        sorted best-first. Defaults status_filter to
        [ClaimStatus.ACTIVE, ClaimStatus.NEEDS_REVIEW] when omitted."""
        ...

    def propose_claim(self, proposal: ClaimProposal, embedding: list[float]) -> Claim:
        """Entry point for extraction pipelines. Resolves the ADR v1
        supersession/conflict policy and commits the outcome atomically."""
        ...

    def get_conflict_group(self, claim_key: str) -> Optional[ConflictGroup]:
        """All needs_review claims sharing `claim_key`, or None if there is
        no open conflict for that key."""
        ...


@runtime_checkable
class FlashRankReranker(Protocol):
    """Cross-encoder reranking boundary, backend-agnostic via the
    `rerankers` library (FlashRank/ms-marco-TinyBERT-L-2-v2 by default)."""

    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Score (query, candidate_text) pairs. Returns one float per pair,
        same order as input, higher is more relevant."""
        ...


@runtime_checkable
class HybridSearcher(Protocol):
    """Top-level retrieval orchestration boundary."""

    async def retrieve(self, query: str) -> list[Candidate]:
        """Run 3-way RRF fusion (vector + FTS + claims), rerank via
        FlashRankReranker, apply the post-rerank status-weight penalty,
        bundle needs_review conflicts, and return at most
        settings.retrieval.retrieval_top_k candidates, best-first."""
        ...
