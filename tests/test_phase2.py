from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from palace.config import Settings, get_settings
from palace.domain.models import ClaimProposal, ClaimStatus, Severity, SourceType
from palace.stores.claims_store import SQLiteClaimsStore


def _vector(position: int) -> list[float]:
    vector = [0.0] * 768
    vector[position] = 1.0
    return vector


def _proposal(
    value: str,
    *,
    entity: str = "person",
    attribute: str = "favorite_color",
    explicit: bool = True,
    severity: Severity = Severity.LOW,
) -> ClaimProposal:
    return ClaimProposal(
        entity=entity,
        attribute=attribute,
        value=value,
        source_type=SourceType.CHAT,
        severity=severity,
        tags=["preference"],
        is_explicit=explicit,
    )


@pytest.fixture
def store(tmp_path: Path) -> SQLiteClaimsStore:
    base = get_settings()
    settings: Settings = base.model_copy(
        update={
            "paths": base.paths.model_copy(
                update={
                    "claims_db_path": str(tmp_path / "claims.db"),
                    "schema_sql_path": str(
                        Path(__file__).parents[1] / "src/palace/stores/schema.sql"
                    ),
                }
            )
        }
    )
    claims_store = SQLiteClaimsStore(settings=settings)
    yield claims_store
    claims_store.close()


def test_fresh_claim_insertion(store: SQLiteClaimsStore) -> None:
    inserted = store.propose_claim(_proposal("blue"), _vector(0))

    assert store.db.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    active = store.get_active_claim("person", "favorite_color")
    assert active is not None
    assert active.id == inserted.id
    assert active.status is ClaimStatus.ACTIVE
    assert active.tags == ["preference"]
    assert store.db.execute("SELECT count(*) FROM vec_claims").fetchone()[0] == 1


def test_explicit_update_atomically_supersedes_and_logs_history(
    store: SQLiteClaimsStore,
) -> None:
    original = store.propose_claim(_proposal("blue"), _vector(0))
    replacement = store.propose_claim(_proposal("green"), _vector(1))

    rows = store.db.execute(
        "SELECT id, status, supersedes_id, superseded_by_id FROM claims ORDER BY created_at"
    ).fetchall()
    by_id = {row["id"]: row for row in rows}
    assert by_id[original.id]["status"] == ClaimStatus.SUPERSEDED.value
    assert by_id[original.id]["superseded_by_id"] == replacement.id
    assert by_id[replacement.id]["status"] == ClaimStatus.ACTIVE.value
    assert by_id[replacement.id]["supersedes_id"] == original.id

    history = store.db.execute("SELECT * FROM claim_history").fetchone()
    assert history["claim_id"] == original.id
    assert history["previous_value"] == "blue"
    assert history["new_value"] == "green"
    assert history["change_reason"] == "explicit_update"
    assert store.db.execute("SELECT count(*) FROM vec_claims").fetchone()[0] == 2


def test_supersession_rolls_back_relational_and_vector_writes_on_failure(
    store: SQLiteClaimsStore,
) -> None:
    original = store.propose_claim(_proposal("blue"), _vector(0))
    store.db.execute(
        """
        CREATE TRIGGER reject_history
        BEFORE INSERT ON claim_history
        BEGIN
            SELECT RAISE(ABORT, 'history write failed');
        END
        """
    )
    store.db.commit()

    with pytest.raises(sqlite3.IntegrityError, match="history write failed"):
        store.propose_claim(_proposal("green"), _vector(1))

    active = store.get_active_claim("person", "favorite_color")
    assert active is not None
    assert active.id == original.id
    assert store.db.execute("SELECT count(*) FROM claims").fetchone()[0] == 1
    assert store.db.execute("SELECT count(*) FROM vec_claims").fetchone()[0] == 1


@pytest.mark.parametrize(
    ("explicit", "severity"),
    [(False, Severity.LOW), (True, Severity.HIGH)],
)
def test_inferred_or_high_severity_update_creates_conflict(
    store: SQLiteClaimsStore,
    explicit: bool,
    severity: Severity,
) -> None:
    original = store.propose_claim(_proposal("blue"), _vector(0))
    conflicting = store.propose_claim(
        _proposal("green", explicit=explicit, severity=severity),
        _vector(1),
    )

    group = store.get_conflict_group("person:favorite_color")
    assert group is not None
    assert set(group.claim_ids) == {original.id, conflicting.id}
    assert all(claim.status is ClaimStatus.NEEDS_REVIEW for claim in group.claims)
    assert store.get_active_claim("person", "favorite_color") is None
    history = store.db.execute("SELECT * FROM claim_history").fetchone()
    assert history["change_reason"] == "inferred_conflict"


def test_sqlite_vec_similarity_search(store: SQLiteClaimsStore) -> None:
    nearest = store.propose_claim(
        _proposal("prefers trains", entity="person", attribute="transport"),
        _vector(0),
    )
    store.propose_claim(
        _proposal("uses dark mode", entity="person", attribute="display"),
        _vector(1),
    )

    results = store.search_claims(_vector(0), "trains", k=1)

    assert [claim.id for claim in results] == [nearest.id]
