"""Presentation-layer attachment of open claim conflicts."""

from __future__ import annotations

from palace.domain.models import Claim, ClaimStatus
from palace.interfaces import ClaimsStoreAdapter
from palace.retrieval.retrieval_types import Candidate


def bundle_conflicts(
    candidates: list[Candidate],
    claims_store: ClaimsStoreAdapter,
) -> list[Candidate]:
    """Attach conflict groups without changing candidate scores or order."""
    for candidate in candidates:
        if candidate.kind != "claim" or not isinstance(candidate.payload, Claim):
            continue
        if candidate.payload.status != ClaimStatus.NEEDS_REVIEW:
            continue
        candidate.conflict_group = claims_store.get_conflict_group(
            candidate.payload.claim_key
        )
    return candidates
