"""Live Phase 2 verification against the local ``data/claims.db`` database.

This module intentionally does not seed or modify claim data. It verifies the
database produced by the manual Phase 2 workflow and therefore expects that
workflow's ``user:employer`` conflict and audit records to already exist.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest
import sqlite_vec

from palace.config import get_settings
from palace.domain.models import ConflictGroup
from palace.interfaces import ClaimsStoreAdapter
from palace.stores.claims_store import SQLiteClaimsStore

DATABASE_PATH = Path(__file__).parents[1] / "data" / "claims.db"


@pytest.fixture(scope="module")
def raw_db() -> Iterator[sqlite3.Connection]:
    assert DATABASE_PATH.is_file(), f"live claims database not found: {DATABASE_PATH}"

    db = sqlite3.connect(DATABASE_PATH)
    db.row_factory = sqlite3.Row

    # Foreign-key enforcement is connection-local in SQLite, so every direct
    # verification connection must enable it just as SQLiteClaimsStore does.
    db.execute("PRAGMA foreign_keys = ON")

    db.enable_load_extension(True)
    try:
        sqlite_vec.load(db)
    finally:
        db.enable_load_extension(False)

    try:
        yield db
    finally:
        db.close()


def test_wal_mode_and_connection_pragmas(raw_db: sqlite3.Connection) -> None:
    assert raw_db.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    assert raw_db.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_sqlite_vec_table_and_raw_distance_query(raw_db: sqlite3.Connection) -> None:
    vector_count = raw_db.execute("SELECT count(*) FROM vec_claims").fetchone()[0]
    claim_count = raw_db.execute("SELECT count(*) FROM claims").fetchone()[0]
    assert vector_count == claim_count
    assert vector_count >= 3

    query_vector = sqlite_vec.serialize_float32(
        [0.0] * get_settings().embedding.dimension
    )
    rows = raw_db.execute(
        """
        SELECT c.value, v.distance
        FROM vec_claims AS v
        JOIN claims AS c ON c.id = v.claim_id
        WHERE v.embedding MATCH ? AND k = 3
        ORDER BY v.distance ASC
        """,
        (query_vector,),
    ).fetchall()

    assert len(rows) == 3
    distances = [row["distance"] for row in rows]
    assert distances == sorted(distances)


def test_conflict_group_contract_against_live_store() -> None:
    store = SQLiteClaimsStore(db_path=DATABASE_PATH)
    adapter: ClaimsStoreAdapter = store
    try:
        assert isinstance(store, ClaimsStoreAdapter)
        group = adapter.get_conflict_group("user:employer")
    finally:
        store.close()

    assert isinstance(group, ConflictGroup)
    assert {claim.value for claim in group.claims} == {"Freelance", "MyStorage"}


def test_history_contains_supersession_and_conflict_events(
    raw_db: sqlite3.Connection,
) -> None:
    reasons = {
        row["change_reason"]
        for row in raw_db.execute("SELECT change_reason FROM claim_history")
    }
    assert {"explicit_update", "inferred_conflict"} <= reasons

