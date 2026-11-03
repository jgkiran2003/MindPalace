"""Resolution pipeline from proposals to persisted claims."""

from __future__ import annotations

from typing import Protocol

from palace.claims.outcomes import ClaimEmbeddingError, ClaimStorageError
from palace.domain.models import Claim, ClaimProposal, ClaimStatus
from palace.interfaces import ClaimsStoreAdapter


class Embedder(Protocol):
    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...


def _preview_claim(proposal: ClaimProposal) -> Claim:
    return Claim(
        entity=proposal.entity,
        attribute=proposal.attribute,
        value=proposal.value,
        status=ClaimStatus.ACTIVE,
        severity=proposal.severity,
        vault_id=proposal.vault_id,
        source_type=proposal.source_type,
        derived_from_chunk_id=proposal.derived_from_chunk_id,
        confidence=proposal.confidence,
        tags=proposal.tags,
    )


def resolve_and_store_proposals(
    proposals: list[ClaimProposal],
    embedder: Embedder,
    claims_store: ClaimsStoreAdapter,
    dry_run: bool = False,
) -> list[Claim]:
    """Preview proposals or batch-embed and resolve them through the store."""
    if dry_run:
        return [_preview_claim(proposal) for proposal in proposals]
    if not proposals:
        return []

    # Embed everything first: an embedding failure must leave the store untouched.
    try:
        embeddings = embedder.embed_batch([proposal.value for proposal in proposals])
        if len(embeddings) != len(proposals):
            raise ValueError("embedder returned a different number of vectors than proposals")
    except Exception as exc:
        raise ClaimEmbeddingError(f"{type(exc).__name__}: {exc}") from exc

    # Each propose_claim is its own transaction, so a mid-batch failure keeps the
    # earlier claims; report them instead of pretending nothing was written.
    persisted: list[Claim] = []
    for proposal, embedding in zip(proposals, embeddings, strict=True):
        try:
            persisted.append(claims_store.propose_claim(proposal, embedding))
        except Exception as exc:
            raise ClaimStorageError(
                f"{type(exc).__name__}: {exc}", persisted
            ) from exc
    return persisted
