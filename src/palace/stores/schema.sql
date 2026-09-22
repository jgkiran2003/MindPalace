-- MindPalace — claims.db schema
-- Applied once, idempotently, by ClaimsStoreAdapter._init_db() on first
-- connection (path comes from settings.paths.schema_sql_path).
-- Requires the sqlite-vec extension to be loaded on the connection BEFORE
-- this script runs (the CREATE VIRTUAL TABLE ... USING vec0 statement
-- fails otherwise).

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS vaults (
    id         TEXT PRIMARY KEY,
    name       TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS claims (
    id                     TEXT PRIMARY KEY,
    entity                 TEXT NOT NULL,
    attribute              TEXT NOT NULL,
    claim_key              TEXT GENERATED ALWAYS AS (entity || ':' || attribute) STORED,
    value                  TEXT NOT NULL,
    status                 TEXT NOT NULL CHECK (status IN ('active', 'superseded', 'needs_review')),
    supersedes_id          TEXT REFERENCES claims(id) ON DELETE SET NULL,
    superseded_by_id       TEXT REFERENCES claims(id) ON DELETE SET NULL,
    severity               TEXT NOT NULL DEFAULT 'low' CHECK (severity IN ('low', 'medium', 'high')),
    vault_id               TEXT REFERENCES vaults(id) ON DELETE SET NULL,
    source_type            TEXT NOT NULL CHECK (source_type IN ('chat', 'document', 'voice', 'email', 'task')),
    derived_from_chunk_id  TEXT,
    confidence             REAL CHECK (confidence IS NULL OR (confidence >= 0.0 AND confidence <= 1.0)),
    created_at             TEXT NOT NULL,
    updated_at             TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_claims_key_status ON claims(claim_key, status);
CREATE INDEX IF NOT EXISTS idx_claims_vault      ON claims(vault_id);
CREATE INDEX IF NOT EXISTS idx_claims_updated_at ON claims(updated_at);

CREATE TABLE IF NOT EXISTS claim_tags (
    claim_id TEXT NOT NULL REFERENCES claims(id) ON DELETE CASCADE,
    tag      TEXT NOT NULL,
    PRIMARY KEY (claim_id, tag)
);
CREATE INDEX IF NOT EXISTS idx_claim_tags_tag ON claim_tags(tag);

CREATE TABLE IF NOT EXISTS claim_references (
    claim_id        TEXT NOT NULL REFERENCES claims(id) ON DELETE CASCADE,
    source_chunk_id TEXT NOT NULL,     -- chunk_id in the LanceDB file store (cross-store, not an FK)
    link_type       TEXT NOT NULL CHECK (link_type IN ('derived_from', 'mentions')),
    PRIMARY KEY (claim_id, source_chunk_id, link_type)
);
CREATE INDEX IF NOT EXISTS idx_claim_refs_chunk ON claim_references(source_chunk_id);

CREATE TABLE IF NOT EXISTS claim_history (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    claim_id        TEXT NOT NULL,     -- intentionally not FK-cascaded: history must survive claim hard-deletion
    previous_value  TEXT,
    new_value       TEXT,
    change_reason   TEXT NOT NULL CHECK (change_reason IN
                     ('explicit_update', 'inferred_conflict', 'compaction', 'manual_review', 'deletion')),
    changed_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_claim_history_claim ON claim_history(claim_id);
CREATE INDEX IF NOT EXISTS idx_claim_history_changed_at ON claim_history(changed_at);

-- Vector index for claim semantic search. Dimension is hardcoded here
-- because sqlite-vec's vec0 module requires a static schema at CREATE
-- time; it MUST match settings.embedding.dimension (768 for the
-- nomic-embed-text default). If the embedding model changes dimension,
-- this table must be dropped and rebuilt.
CREATE VIRTUAL TABLE IF NOT EXISTS vec_claims USING vec0(
    claim_id   TEXT PRIMARY KEY,
    embedding  FLOAT[768]
);
