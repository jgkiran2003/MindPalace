"""Transactional SQLite and sqlite-vec implementation of the claims store."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import sqlite_vec

from palace.config import Settings, get_settings
from palace.domain.models import (
    ChangeReason,
    Claim,
    ClaimProposal,
    ClaimStatus,
    ConflictGroup,
    Severity,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SQLiteClaimsStore:
    """SQLite-backed claims adapter with vectors committed atomically."""

    def __init__(
        self,
        db_path: str | Path | None = None,
        *,
        settings: Settings | None = None,
        schema_path: str | Path | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        resolved_db_path = str(db_path or self.settings.paths.claims_db_path)
        if resolved_db_path != ":memory:":
            Path(resolved_db_path).parent.mkdir(parents=True, exist_ok=True)

        self.db = sqlite3.connect(resolved_db_path)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys = ON")
        self.db.execute("PRAGMA journal_mode = WAL")
        self._load_sqlite_vec()
        self._init_db(Path(schema_path or self.settings.paths.schema_sql_path))

    def _load_sqlite_vec(self) -> None:
        self.db.enable_load_extension(True)
        try:
            sqlite_vec.load(self.db)
        finally:
            self.db.enable_load_extension(False)

    def _init_db(self, schema_path: Path) -> None:
        self.db.executescript(schema_path.read_text(encoding="utf-8"))
        self.db.commit()

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        self.db.execute("BEGIN IMMEDIATE")
        try:
            yield
        except BaseException:
            self.db.rollback()
            raise
        else:
            self.db.commit()

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> "SQLiteClaimsStore":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def _validate_embedding(self, embedding: list[float]) -> None:
        expected = self.settings.embedding.dimension
        if len(embedding) != expected:
            raise ValueError(f"embedding must have {expected} dimensions, got {len(embedding)}")

    def _claim_from_row(self, row: sqlite3.Row) -> Claim:
        tags = [
            tag_row["tag"]
            for tag_row in self.db.execute(
                "SELECT tag FROM claim_tags WHERE claim_id = ? ORDER BY tag",
                (row["id"],),
            )
        ]
        return Claim(
            id=row["id"],
            entity=row["entity"],
            attribute=row["attribute"],
            value=row["value"],
            status=row["status"],
            supersedes_id=row["supersedes_id"],
            superseded_by_id=row["superseded_by_id"],
            severity=row["severity"],
            vault_id=row["vault_id"],
            source_type=row["source_type"],
            derived_from_chunk_id=row["derived_from_chunk_id"],
            confidence=row["confidence"],
            tags=tags,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def _new_claim(
        self,
        proposal: ClaimProposal,
        status: ClaimStatus,
        *,
        supersedes_id: str | None = None,
    ) -> Claim:
        return Claim(
            entity=proposal.entity,
            attribute=proposal.attribute,
            value=proposal.value,
            status=status,
            supersedes_id=supersedes_id,
            severity=proposal.severity,
            vault_id=proposal.vault_id,
            source_type=proposal.source_type,
            derived_from_chunk_id=proposal.derived_from_chunk_id,
            confidence=proposal.confidence,
            tags=list(dict.fromkeys(proposal.tags)),
        )

    def _insert_claim(self, claim: Claim, embedding: list[float]) -> None:
        self.db.execute(
            """
            INSERT INTO claims (
                id, entity, attribute, value, status, supersedes_id,
                superseded_by_id, severity, vault_id, source_type,
                derived_from_chunk_id, confidence, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                claim.id,
                claim.entity,
                claim.attribute,
                claim.value,
                claim.status.value,
                claim.supersedes_id,
                claim.superseded_by_id,
                claim.severity.value,
                claim.vault_id,
                claim.source_type.value,
                claim.derived_from_chunk_id,
                claim.confidence,
                claim.created_at,
                claim.updated_at,
            ),
        )
        self.db.executemany(
            "INSERT INTO claim_tags (claim_id, tag) VALUES (?, ?)",
            [(claim.id, tag) for tag in claim.tags],
        )
        self.db.execute(
            "INSERT INTO vec_claims (claim_id, embedding) VALUES (?, ?)",
            (claim.id, sqlite_vec.serialize_float32(embedding)),
        )

    def _active_claims_for_key(self, claim_key: str) -> list[Claim]:
        rows = self.db.execute(
            """
            SELECT * FROM claims
            WHERE claim_key = ? AND status = ?
            ORDER BY updated_at DESC, id ASC
            """,
            (claim_key, ClaimStatus.ACTIVE.value),
        ).fetchall()
        return [self._claim_from_row(row) for row in rows]

    def _insert_history(
        self,
        existing: Claim,
        new_value: str,
        reason: ChangeReason,
        changed_at: str,
    ) -> None:
        self.db.execute(
            """
            INSERT INTO claim_history (
                claim_id, previous_value, new_value, change_reason, changed_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (existing.id, existing.value, new_value, reason.value, changed_at),
        )

    def propose_claim(self, proposal: ClaimProposal, embedding: list[float]) -> Claim:
        self._validate_embedding(embedding)

        # Construct once up front so Claim's locked validation/normalization is
        # also used to form the generated-column lookup key.
        candidate = self._new_claim(proposal, ClaimStatus.ACTIVE)
        with self._transaction():
            active_claims = self._active_claims_for_key(candidate.claim_key)
            if not active_claims:
                self._insert_claim(candidate, embedding)
                return candidate

            blocks_auto_resolve = (
                self.settings.claims.high_severity_blocks_auto_resolve
                and proposal.severity == Severity.HIGH
            )
            if proposal.is_explicit and not blocks_auto_resolve:
                existing = active_claims[0]
                replacement = self._new_claim(
                    proposal,
                    ClaimStatus.ACTIVE,
                    supersedes_id=existing.id,
                )
                # superseded_by_id is an immediate foreign key, so the new
                # row must exist before the old row can point at it. Both
                # writes remain inside the same transaction.
                self._insert_claim(replacement, embedding)
                self.db.execute(
                    """
                    UPDATE claims
                    SET status = ?, superseded_by_id = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        ClaimStatus.SUPERSEDED.value,
                        replacement.id,
                        replacement.updated_at,
                        existing.id,
                    ),
                )
                self._insert_history(
                    existing,
                    replacement.value,
                    ChangeReason.EXPLICIT_UPDATE,
                    replacement.updated_at,
                )
                return replacement

            conflict = self._new_claim(proposal, ClaimStatus.NEEDS_REVIEW)
            for existing in active_claims:
                self.db.execute(
                    "UPDATE claims SET status = ?, updated_at = ? WHERE id = ?",
                    (ClaimStatus.NEEDS_REVIEW.value, conflict.updated_at, existing.id),
                )
                self._insert_history(
                    existing,
                    conflict.value,
                    ChangeReason.INFERRED_CONFLICT,
                    conflict.updated_at,
                )
            self._insert_claim(conflict, embedding)
            return conflict

    def get_active_claim(self, entity: str, attribute: str) -> Optional[Claim]:
        row = self.db.execute(
            """
            SELECT * FROM claims
            WHERE claim_key = ? AND status = ?
            ORDER BY updated_at DESC, id ASC
            LIMIT 1
            """,
            (f"{entity.strip()}:{attribute.strip()}", ClaimStatus.ACTIVE.value),
        ).fetchone()
        return self._claim_from_row(row) if row is not None else None

    def get_conflict_group(self, claim_key: str) -> Optional[ConflictGroup]:
        rows = self.db.execute(
            """
            SELECT * FROM claims
            WHERE claim_key = ? AND status = ?
            ORDER BY created_at ASC, id ASC
            """,
            (claim_key, ClaimStatus.NEEDS_REVIEW.value),
        ).fetchall()
        if len(rows) < 2:
            return None
        claims = [self._claim_from_row(row) for row in rows]
        return ConflictGroup(
            claim_key=claim_key,
            claims=claims,
            opened_at=min(claim.created_at for claim in claims),
        )

    def search_claims(
        self,
        query_embedding: list[float],
        query_text: str,
        k: int,
        status_filter: Optional[list[ClaimStatus]] = None,
    ) -> list[Claim]:
        self._validate_embedding(query_embedding)
        if k <= 0:
            return []
        statuses = (
            [ClaimStatus.ACTIVE, ClaimStatus.NEEDS_REVIEW]
            if status_filter is None
            else status_filter
        )
        if not statuses:
            return []

        vector_count = self.db.execute("SELECT count(*) FROM vec_claims").fetchone()[0]
        if vector_count == 0:
            return []

        placeholders = ", ".join("?" for _ in statuses)
        sql = f"""
            WITH nearest AS (
                SELECT claim_id, distance
                FROM vec_claims
                WHERE embedding MATCH ? AND k = ?
            )
            SELECT claims.*
            FROM nearest
            JOIN claims ON claims.id = nearest.claim_id
            WHERE claims.status IN ({placeholders})
            ORDER BY nearest.distance ASC, claims.id ASC
            LIMIT ?
        """
        parameters: list[Any] = [
            sqlite_vec.serialize_float32(query_embedding),
            vector_count,
            *(status.value for status in statuses),
            k,
        ]
        rows = self.db.execute(sql, parameters).fetchall()
        # Phase 2 is vector-only; query_text becomes the lexical leg in Phase 3.
        _ = query_text
        return [self._claim_from_row(row) for row in rows]

    def list_claims(self, status: ClaimStatus | None = None) -> list[Claim]:
        if status is None:
            rows = self.db.execute(
                "SELECT * FROM claims ORDER BY updated_at DESC, id ASC"
            ).fetchall()
        else:
            rows = self.db.execute(
                """
                SELECT * FROM claims
                WHERE status = ?
                ORDER BY updated_at DESC, id ASC
                """,
                (status.value,),
            ).fetchall()
        return [self._claim_from_row(row) for row in rows]


# A concise public name for callers while retaining the backend-specific name.
ClaimsStore = SQLiteClaimsStore
