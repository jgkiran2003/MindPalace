"""
MindPalace — LanceDB chunk table schema
The exact PyArrow schema for the document-chunk table. Field names and
types here must mirror domain.models.DocumentChunk field-for-field —
stores/file_store.py converts between the two on every read/write.
"""

from __future__ import annotations

import pyarrow as pa

from palace.config import get_settings

_EMBEDDING_DIM = get_settings().embedding.dimension  # 768 for nomic-embed-text (default)


def build_chunk_schema() -> pa.Schema:
    """Called once by file_store.py when creating (or verifying) the table."""
    return pa.schema(
        [
            pa.field("chunk_id", pa.string(), nullable=False),
            pa.field("doc_path", pa.string(), nullable=False),
            pa.field("chunk_index", pa.int32(), nullable=False),
            pa.field("text", pa.string(), nullable=False),
            pa.field("vector", pa.list_(pa.float32(), _EMBEDDING_DIM), nullable=False),
            pa.field("status", pa.string(), nullable=False),  # 'raw' | 'compiled'
            pa.field("derived_from_chunk_id", pa.string(), nullable=True),
            pa.field("tags", pa.list_(pa.string()), nullable=False),
            pa.field("severity", pa.string(), nullable=False),  # 'low' | 'medium' | 'high'
            pa.field("vault_id", pa.string(), nullable=True),
            # 'chat' | 'document' | 'voice' | 'email' | 'task'
            pa.field("source_type", pa.string(), nullable=False),
            pa.field("created_at", pa.timestamp("us", tz="UTC"), nullable=False),
            pa.field("updated_at", pa.timestamp("us", tz="UTC"), nullable=False),
        ]
    )


CHUNK_TABLE_NAME = "document_chunks"
VECTOR_COLUMN = "vector"
FTS_COLUMN = "text"
