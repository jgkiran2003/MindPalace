"""Hermetic integration checks for persistent MindPalace invariants."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest
import sqlite_vec

from palace.domain.models import ClaimProposal, ConflictGroup, SourceType
from palace.interfaces import ClaimsStoreAdapter
from palace.stores.claims_store import SQLiteClaimsStore

SCHEMA_PATH = Path(__file__).parents[1] / "src/palace/stores/schema.sql"


def _vector(value: float = 0.1) -> list[float]:
    return [value] * 768


def _proposal(
    value: str,
    *,
    attribute: str = "employer",
    explicit: bool,
) -> ClaimProposal:
    return ClaimProposal(
        entity="user",
        attribute=attribute,
        value=value,
        source_type=SourceType.CHAT,
        is_explicit=explicit,
    )


@pytest.fixture
def isolated_store(
    tmp_path: Path,
) -> Iterator[tuple[SQLiteClaimsStore, Path]]:
    db_path = tmp_path / "claims.db"
    store = SQLiteClaimsStore(db_path=db_path, schema_path=SCHEMA_PATH)
    try:
        yield store, db_path
    finally:
        store.close()


def _load_sqlite_vec(db: sqlite3.Connection) -> None:
    db.enable_load_extension(True)
    try:
        sqlite_vec.load(db)
    finally:
        db.enable_load_extension(False)


def test_wal_mode_and_foreign_keys(
    isolated_store: tuple[SQLiteClaimsStore, Path],
) -> None:
    store, db_path = isolated_store
    assert store.db.execute("PRAGMA foreign_keys").fetchone()[0] == 1

    with sqlite3.connect(db_path) as db:
        # foreign_keys is connection-local and must be enabled on every raw
        # connection; WAL mode persists on the database itself.
        db.execute("PRAGMA foreign_keys = ON")
        assert db.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        assert db.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_sqlite_vec_native_knn_query(
    isolated_store: tuple[SQLiteClaimsStore, Path],
) -> None:
    store, db_path = isolated_store
    vector = _vector()
    store.propose_claim(
        _proposal("MyStorage", explicit=True),
        vector,
    )

    with sqlite3.connect(db_path) as db:
        _load_sqlite_vec(db)
        claim_count = db.execute("SELECT count(*) FROM claims").fetchone()[0]
        vector_count = db.execute("SELECT count(*) FROM vec_claims").fetchone()[0]
        assert vector_count == claim_count == 1

        rows = db.execute(
            """
            SELECT c.value, v.distance
            FROM vec_claims AS v
            JOIN claims AS c ON c.id = v.claim_id
            WHERE v.embedding MATCH ? AND k = 1
            ORDER BY v.distance ASC
            """,
            (sqlite_vec.serialize_float32(vector),),
        ).fetchall()

    assert len(rows) == 1
    assert rows[0][0] == "MyStorage"
    assert rows[0][1] == pytest.approx(0.0)


def test_conflict_group_contract(
    isolated_store: tuple[SQLiteClaimsStore, Path],
) -> None:
    store, _ = isolated_store
    adapter: ClaimsStoreAdapter = store
    store.propose_claim(
        _proposal("MyStorage", explicit=True),
        _vector(0.1),
    )
    store.propose_claim(
        _proposal("Freelance", explicit=False),
        _vector(0.2),
    )

    group = adapter.get_conflict_group("user:employer")

    assert isinstance(group, ConflictGroup)
    assert {claim.value for claim in group.claims} == {"Freelance", "MyStorage"}


def test_audit_trail_records_updates_and_conflicts(
    isolated_store: tuple[SQLiteClaimsStore, Path],
) -> None:
    store, _ = isolated_store
    store.propose_claim(
        _proposal("Engineer", attribute="role", explicit=True),
        _vector(0.1),
    )
    store.propose_claim(
        _proposal("Lead", attribute="role", explicit=True),
        _vector(0.2),
    )
    store.propose_claim(
        _proposal("Advisor", attribute="role", explicit=False),
        _vector(0.3),
    )

    reasons = {
        row[0]
        for row in store.db.execute(
            "SELECT change_reason FROM claim_history"
        ).fetchall()
    }
    assert {"explicit_update", "inferred_conflict"} <= reasons
