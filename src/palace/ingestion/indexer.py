"""Vault scanner and embedding/indexing orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from palace.config import Settings, get_settings
from palace.ingestion.chunker import MarkdownChunker
from palace.llm.embedder import OllamaEmbedder
from palace.stores.file_store import LanceFileStore


@dataclass(frozen=True)
class IndexStats:
    documents: int = 0
    chunks: int = 0


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

    def index(self) -> IndexStats:
        vault = Path(self.settings.paths.vault_dir)
        paths = sorted(vault.rglob("*.md")) if vault.exists() else []
        write_count = 0
        total_chunks = 0

        for path in paths:
            chunks = self.chunker.chunk(path)
            self.store.delete_chunks_by_doc_path(str(path))
            batch_size = self.settings.embedding.batch_size
            for offset in range(0, len(chunks), batch_size):
                batch = chunks[offset : offset + batch_size]
                vectors = self.embedder.embed_many(chunk.text for chunk in batch)
                embedded = [chunk.model_copy(update={"vector": vector}) for chunk, vector in zip(batch, vectors, strict=True)]
                cursor = 0
                optimize_after = self.settings.indexing.optimize_after_writes
                while cursor < len(embedded):
                    until_optimize = optimize_after - write_count
                    write_batch = embedded[cursor : cursor + until_optimize]
                    self.store.upsert_chunks(write_batch)
                    cursor += len(write_batch)
                    write_count += len(write_batch)
                    total_chunks += len(write_batch)
                    if write_count == optimize_after:
                        self.store.optimize()
                        write_count = 0

        # Make the final partial batch visible to the native FTS index.
        if write_count:
            self.store.optimize()
        return IndexStats(documents=len(paths), chunks=total_chunks)


def index_vault(settings: Settings | None = None) -> IndexStats:
    return VaultIndexer(settings=settings).index()
