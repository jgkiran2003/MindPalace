"""
MindPalace — Core Domain Models
Pydantic v2 models shared across the claims store and the file store.
This is the one file both `stores/claims_store.py` and `stores/file_store.py`
import from — no module redefines these shapes locally.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from palace.config import get_settings


def _now() -> str:
    """ISO-8601 UTC timestamp, matching every TEXT timestamp column in claims.db
    and every timestamp("us", tz="UTC") field in the LanceDB chunk schema."""
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    return str(uuid.uuid4())


# --------------------------------------------------------------------------- #
# Shared enums
# --------------------------------------------------------------------------- #


class ClaimStatus(str, Enum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    NEEDS_REVIEW = "needs_review"


class ChunkStatus(str, Enum):
    RAW = "raw"
    COMPILED = "compiled"


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class SourceType(str, Enum):
    CHAT = "chat"
    DOCUMENT = "document"
    VOICE = "voice"
    EMAIL = "email"
    TASK = "task"


class ChangeReason(str, Enum):
    EXPLICIT_UPDATE = "explicit_update"
    INFERRED_CONFLICT = "inferred_conflict"
    COMPACTION = "compaction"
    MANUAL_REVIEW = "manual_review"
    DELETION = "deletion"


# --------------------------------------------------------------------------- #
# Claims-store models
# --------------------------------------------------------------------------- #


class Claim(BaseModel):
    """A single atomic, revisable assertion — the row shape of the SQLite
    `claims` table (see stores/schema.sql)."""

    id: str = Field(default_factory=_new_id)
    entity: str
    attribute: str
    value: str
    status: ClaimStatus = ClaimStatus.ACTIVE
    supersedes_id: Optional[str] = None
    superseded_by_id: Optional[str] = None
    severity: Severity = Severity.LOW
    vault_id: Optional[str] = None
    source_type: SourceType
    derived_from_chunk_id: Optional[str] = None
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    tags: list[str] = Field(default_factory=list)  # joined from claim_tags at read time
    created_at: str = Field(default_factory=_now)
    updated_at: str = Field(default_factory=_now)

    @property
    def claim_key(self) -> str:
        """Matches the SQLite GENERATED column `claims.claim_key` — the
        join key for supersession lookups and conflict grouping."""
        return f"{self.entity}:{self.attribute}"

    @field_validator("entity", "attribute")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("entity/attribute must be non-empty")
        return v.strip()


class ClaimProposal(BaseModel):
    """
    What claims/extraction.py produces BEFORE it touches the store. Not yet
    a Claim — no id, no status, no conflict-resolution decision made.
    claims/resolver.py -> ClaimsStoreAdapter.propose_claim() turns this into
    a committed Claim (auto-resolved or parked in needs_review).
    """

    entity: str
    attribute: str
    value: str
    source_type: SourceType
    severity: Severity = Severity.LOW
    vault_id: Optional[str] = None
    derived_from_chunk_id: Optional[str] = None
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    tags: list[str] = Field(default_factory=list)
    is_explicit: bool = False
    """
    True when the source text is an unambiguous first-person statement (the
    "structural match" resolution rule from ADR v1, Area 2). Set by
    claims/extraction.py, consumed by propose_claim() to decide eager
    auto-supersede vs. needs_review. Auto-supersede additionally requires
    `severity != Severity.HIGH` when
    settings.claims.high_severity_blocks_auto_resolve is true.
    """


class ConflictGroup(BaseModel):
    """
    A bundle of needs_review claims sharing one claim_key — the unit the
    Conflicts-queue UI renders and retrieval/conflict_bundler.py attaches
    to search results. Returned by
    ClaimsStoreAdapter.get_conflict_group(); None if there is no open
    conflict for that key.
    """

    claim_key: str
    claims: list[Claim] = Field(min_length=1)
    opened_at: str = Field(default_factory=_now)  # earliest member claim's created_at

    @model_validator(mode="after")
    def _validate_group(self) -> "ConflictGroup":
        for claim in self.claims:
            if claim.status != ClaimStatus.NEEDS_REVIEW:
                raise ValueError("ConflictGroup can only contain needs_review claims")
            if claim.claim_key != self.claim_key:
                raise ValueError(
                    f"claim {claim.id} has claim_key {claim.claim_key!r}, "
                    f"expected {self.claim_key!r}"
                )
        return self

    @property
    def claim_ids(self) -> list[str]:
        return [c.id for c in self.claims]


class Vault(BaseModel):
    id: str = Field(default_factory=_new_id)
    name: str
    created_at: str = Field(default_factory=_now)


class ClaimHistoryEntry(BaseModel):
    id: Optional[int] = None  # autoincrement, unset until read back from claim_history
    claim_id: str
    previous_value: Optional[str] = None
    new_value: Optional[str] = None
    change_reason: ChangeReason
    changed_at: str = Field(default_factory=_now)


# --------------------------------------------------------------------------- #
# File-store model
# --------------------------------------------------------------------------- #


class DocumentChunk(BaseModel):
    """
    The Python-side mirror of one row in the LanceDB chunk table (see
    stores/lance_schema.py for the exact PyArrow schema this must match
    field-for-field). Produced by ingestion/chunker.py, consumed by
    stores/file_store.py and retrieval/hybrid_searcher.py.
    """

    chunk_id: str = Field(default_factory=_new_id)
    doc_path: str
    chunk_index: int = Field(ge=0)
    text: str
    vector: list[float]
    status: ChunkStatus = ChunkStatus.COMPILED
    derived_from_chunk_id: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    severity: Severity = Severity.LOW
    vault_id: Optional[str] = None
    source_type: SourceType
    created_at: str = Field(default_factory=_now)
    updated_at: str = Field(default_factory=_now)

    @field_validator("vector")
    @classmethod
    def _check_dimension(cls, v: list[float]) -> list[float]:
        expected = get_settings().embedding.dimension
        if len(v) != expected:
            raise ValueError(f"vector must have {expected} dimensions (settings.embedding.dimension), got {len(v)}")
        return v
