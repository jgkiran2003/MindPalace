"""
MindPalace — Shared retrieval types
Extracted from hybrid_searcher.py so interfaces.py can reference `Candidate`
without importing the concrete HybridSearcher implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

from palace.domain.models import Claim, ConflictGroup, DocumentChunk


@dataclass
class Candidate:
    """
    Unified shape for a document chunk OR a claim, so both can be fused,
    reranked, and scored with the same code path in HybridSearcher.
    """

    id: str
    text: str  # what gets fed to FlashRankReranker.score()
    kind: Literal["chunk", "claim"]
    status: str  # ChunkStatus value for chunks, ClaimStatus value for claims
    payload: DocumentChunk | Claim
    rrf_score: float = 0.0
    flashrank_score: float = 0.0
    final_score: float = 0.0
    conflict_group: Optional[ConflictGroup] = None
