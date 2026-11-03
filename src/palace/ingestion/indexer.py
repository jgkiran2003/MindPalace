"""Vault scanner and embedding/indexing orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from palace.config import Settings, get_settings
from palace.domain.models import DocumentChunk
from palace.ingestion.chunker import MarkdownChunker, is_archived_note
from palace.llm.embedder import OllamaEmbedder
from palace.stores.file_store import LanceFileStore


@dataclass(frozen=True)
class IndexStats:
    documents: int = 0
    chunks: int = 0
    archived: int = 0


class VaultIndexer:
    def __init__(
        self,
        *,
        settings: Settings | None = None,
        store: LanceFileStore | None = None,
        embedder: OllamaEmbedder | None = None,
        chunker: MarkdownChunker | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.store = store or LanceFileStore(settings=self.settings)
        self.embedder = embedder or OllamaEmbedder(settings=self.settings)
        self.chunker = chunker or MarkdownChunker(settings=self.settings)
        self._write_count = 0

    def _embed(self, chunks: list[DocumentChunk]) -> list[DocumentChunk]:
        batch_size = self.settings.embedding.batch_size
        embedded: list[DocumentChunk] = []
        for offset in range(0, len(chunks), batch_size):
            batch = chunks[offset : offset + batch_size]
            vectors = self.embedder.embed_many(chunk.text for chunk in batch)
            embedded.extend(
                chunk.model_copy(update={"vector": vector})
                for chunk, vector in zip(batch, vectors, strict=True)
            )
        return embedded

    def _write(self, chunks: list[DocumentChunk]) -> None:
        optimize_after = self.settings.indexing.optimize_after_writes
        cursor = 0
        while cursor < len(chunks):
            until_optimize = optimize_after - self._write_count
            write_batch = chunks[cursor : cursor + until_optimize]
            self.store.upsert_chunks(write_batch)
            cursor += len(write_batch)
            self._write_count += len(write_batch)
            if self._write_count == optimize_after:
                self.store.optimize()
                self._write_count = 0

    def _flush(self) -> None:
        # Make the final partial batch visible to the native FTS index.
        if self._write_count:
            self.store.optimize()
            self._write_count = 0

    def _index_path(self, path: Path) -> tuple[int, bool]:
        """Replace one file's chunks; returns (chunks written, archived?)."""
        if is_archived_note(path):
            self.store.delete_chunks_by_doc_path(str(path))
            return 0, True
        # Chunk and embed before touching the store so a failure here leaves
        # the document's existing chunks intact.
        embedded = self._embed(self.chunker.chunk(path))
        self.store.delete_chunks_by_doc_path(str(path))
        self._write(embedded)
        return len(embedded), False

    def index_file(self, path: str | Path) -> int:
        """Index (or, for archived notes, de-index) a single vault file."""
        try:
            written, _ = self._index_path(Path(path))
        finally:
            self._flush()
        return written

    def remove_file(self, path: str | Path) -> int:
        """Drop a file's chunks from the index (the vault file is not touched)."""
        return self.store.delete_chunks_by_doc_path(str(path))

    def index(self) -> IndexStats:
        vault = Path(self.settings.paths.vault_dir)
        paths = sorted(vault.rglob("*.md")) if vault.exists() else []
        total_chunks = 0
        archived = 0

        try:
            for path in paths:
                written, was_archived = self._index_path(path)
                total_chunks += written
                archived += int(was_archived)
        finally:
            self._flush()
        return IndexStats(documents=len(paths), chunks=total_chunks, archived=archived)


def index_vault(settings: Settings | None = None) -> IndexStats:
    return VaultIndexer(settings=settings).index()
