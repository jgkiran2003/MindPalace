from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import pytest

from palace.cli import rerank_and_penalize
from palace.domain.models import ChunkStatus, DocumentChunk, Severity, SourceType
from palace.ingestion.chunker import chunk_markdown
from palace.retrieval.retrieval_types import Candidate
from palace.stores.file_store import LanceFileStore


def _vector(position: int) -> list[float]:
    vector = [0.0] * 768
    vector[position] = 1.0
    return vector


def _chunk(
    chunk_id: str,
    text: str,
    vector: list[float],
    *,
    status: ChunkStatus = ChunkStatus.COMPILED,
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        doc_path="note.md",
        chunk_index=0,
        text=text,
        vector=vector,
        status=status,
        source_type=SourceType.DOCUMENT,
    )


def test_chunk_markdown_parses_frontmatter_and_heading_sections(tmp_path: Path) -> None:
    note = tmp_path / "note.md"
    note.write_text(
        """---
status: raw
tags: [memory, retrieval]
severity: medium
source_type: document
vault_id: personal
---
# Mind Palace
The first paragraph belongs to the top-level section.

This is a second paragraph in the same heading.

## Search
Hybrid search combines vector similarity and full-text search.
""",
        encoding="utf-8",
    )

    chunks = chunk_markdown(note)

    assert len(chunks) == 3
    assert chunks[0].text == "# Mind Palace\n\nThe first paragraph belongs to the top-level section."
    assert chunks[1].text == "# Mind Palace\n\nThis is a second paragraph in the same heading."
    assert chunks[2].text.startswith("# Mind Palace\n## Search")
    assert chunks[2].chunk_index == 2
    assert chunks[2].status is ChunkStatus.RAW
    assert chunks[2].severity is Severity.MEDIUM
    assert chunks[2].tags == ["memory", "retrieval"]
    assert chunks[2].vault_id == "personal"
    assert len(chunks[2].vector) == 768


def test_lancedb_vector_and_native_fts_retrieval(tmp_path: Path) -> None:
    store = LanceFileStore(tmp_path / "lance")
    vector_match = _chunk("vector-match", "A note about semantic astronomy", _vector(0))
    text_match = _chunk("text-match", "Alpacas prefer searchable mountain notes", _vector(1))
    store.upsert_chunks([vector_match, text_match])
    store.optimize()

    vector_results = store.vector_search(_vector(0), limit=2)
    fts_results = store.fts_search("alpacas", limit=2)

    assert vector_results[0].chunk_id == "vector-match"
    assert [chunk.chunk_id for chunk in fts_results] == ["text-match"]
    assert any(index.index_type == "FTS" for index in store.table.list_indices())


class _FixedReranker:
    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
        assert len(pairs) == 2
        return [0.9, 0.8]


def test_raw_status_penalty_is_applied_after_flashrank_scoring() -> None:
    raw = _chunk("raw", "raw candidate", _vector(0), status=ChunkStatus.RAW)
    compiled = _chunk("compiled", "compiled candidate", _vector(1))
    candidates = [
        Candidate(id=raw.chunk_id, text=raw.text, kind="chunk", status=raw.status.value, payload=raw),
        Candidate(
            id=compiled.chunk_id,
            text=compiled.text,
            kind="chunk",
            status=compiled.status.value,
            payload=compiled,
        ),
    ]

    results = rerank_and_penalize(
        "candidate",
        candidates,
        _FixedReranker(),  # type: ignore[arg-type]
        raw_status_penalty=0.4,
        limit=2,
    )

    by_id = {candidate.id: candidate for candidate in results}
    assert by_id["raw"].flashrank_score == pytest.approx(0.9)
    assert by_id["raw"].final_score == pytest.approx(0.9 * 0.4)
    assert by_id["compiled"].flashrank_score == pytest.approx(0.8)
    assert by_id["compiled"].final_score == pytest.approx(0.8)
    assert [candidate.id for candidate in results] == ["compiled", "raw"]
