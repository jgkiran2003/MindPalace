"""LanceDB implementation of the locked ``FileStore`` protocol."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lancedb
import pyarrow as pa

from palace.config import Settings, get_settings
from palace.domain.models import DocumentChunk
from palace.stores.lance_schema import CHUNK_TABLE_NAME, FTS_COLUMN, VECTOR_COLUMN, build_chunk_schema


def _timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _row_from_chunk(chunk: DocumentChunk) -> dict[str, Any]:
    row = chunk.model_dump(mode="json")
    row["created_at"] = _timestamp(chunk.created_at)
    row["updated_at"] = _timestamp(chunk.updated_at)
    return row


def _chunk_from_row(row: dict[str, Any]) -> DocumentChunk:
    cleaned = {key: value for key, value in row.items() if not key.startswith("_")}
    for field in ("created_at", "updated_at"):
        value = cleaned[field]
        if isinstance(value, datetime):
            cleaned[field] = value.isoformat()
    vector = cleaned.get("vector")
    if hasattr(vector, "tolist"):
        cleaned["vector"] = vector.tolist()
    return DocumentChunk.model_validate(cleaned)


class LanceFileStore:
    """Persistent, idempotent chunk storage backed by a Lance table."""

    def __init__(self, lance_dir: str | Path | None = None, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        location = Path(lance_dir or self.settings.paths.lance_dir)
        location.mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(str(location))
        table_names = self.db.table_names()
        if CHUNK_TABLE_NAME in table_names:
            self.table = self.db.open_table(CHUNK_TABLE_NAME)
        else:
            self.table = self.db.create_table(CHUNK_TABLE_NAME, schema=build_chunk_schema())
        self._ensure_fts_index()

    def _ensure_fts_index(self) -> None:
        # LanceDB 0.39 can stall while constructing FTS for an empty table.
        # Defer construction until the first rows have been committed.
        if self.table.count_rows() == 0:
            return
        names = {getattr(index, "name", "") for index in self.table.list_indices()}
        if not any(FTS_COLUMN in name for name in names):
            # Deliberately use Lance native FTS; no deprecated Tantivy flags.
            self.table.create_fts_index(FTS_COLUMN)

    def upsert_chunks(self, chunks: list[DocumentChunk]) -> None:
        if not chunks:
            return
        data = pa.Table.from_pylist([_row_from_chunk(chunk) for chunk in chunks], schema=build_chunk_schema())
        (
            self.table.merge_insert("chunk_id")
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .execute(data)
        )
        self._ensure_fts_index()

    def delete_chunks_by_doc_path(self, doc_path: str) -> int:
        escaped = doc_path.replace("'", "''")
        predicate = f"doc_path = '{escaped}'"
        count = self.table.count_rows(predicate)
        if count:
            self.table.delete(predicate)
        return count

    def vector_search(self, query_vector: list[float], limit: int) -> list[DocumentChunk]:
        rows = (
            self.table.search(query_vector, vector_column_name=VECTOR_COLUMN, query_type="vector")
            .limit(limit)
            .to_arrow()
            .to_pylist()
        )
        return [_chunk_from_row(row) for row in rows]

    def fts_search(self, query_text: str, limit: int) -> list[DocumentChunk]:
        if not query_text.strip():
            return []
        if self.table.count_rows() == 0:
            return []
        self._ensure_fts_index()
        rows = (
            self.table.search(query_text, query_type="fts", fts_columns=FTS_COLUMN)
            .limit(limit)
            .to_arrow()
            .to_pylist()
        )
        return [_chunk_from_row(row) for row in rows]

    def optimize(self) -> None:
        self._ensure_fts_index()
        self.table.optimize()
