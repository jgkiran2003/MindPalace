"""Phase-1 command line interface for indexing and hybrid retrieval."""

from __future__ import annotations

import click

from palace.config import Settings, get_settings
from palace.domain.models import ChunkStatus, DocumentChunk
from palace.ingestion.indexer import index_vault
from palace.llm.embedder import OllamaEmbedder
from palace.retrieval.reranker import FlashRankReranker
from palace.retrieval.retrieval_types import Candidate
from palace.stores.file_store import LanceFileStore


def fuse_rrf(
    vector_results: list[DocumentChunk],
    fts_results: list[DocumentChunk],
    *,
    rrf_k: int,
    limit: int,
) -> list[Candidate]:
    """Fuse vector and FTS ranks using the specification's 1-indexed RRF."""
    by_id: dict[str, Candidate] = {}
    for results in (vector_results, fts_results):
        for rank, chunk in enumerate(results, start=1):
            candidate = by_id.setdefault(
                chunk.chunk_id,
                Candidate(
                    id=chunk.chunk_id,
                    text=chunk.text,
                    kind="chunk",
                    status=chunk.status.value,
                    payload=chunk,
                ),
            )
            candidate.rrf_score += 1.0 / (rrf_k + rank)
    return sorted(by_id.values(), key=lambda candidate: (-candidate.rrf_score, candidate.id))[:limit]


def rerank_and_penalize(
    query: str,
    candidates: list[Candidate],
    reranker: FlashRankReranker,
    *,
    raw_status_penalty: float,
    limit: int,
) -> list[Candidate]:
    """Rerank first, then apply the raw-note status multiplier."""
    scores = reranker.score([(query, candidate.text) for candidate in candidates])
    if len(scores) != len(candidates):
        raise ValueError("reranker returned a different number of scores than candidates")
    for candidate, score in zip(candidates, scores, strict=True):
        candidate.flashrank_score = float(score)
        weight = raw_status_penalty if candidate.status == ChunkStatus.RAW.value else 1.0
        candidate.final_score = candidate.flashrank_score * weight
    return sorted(candidates, key=lambda candidate: (-candidate.final_score, candidate.id))[:limit]


def search_chunks(
    query: str,
    *,
    settings: Settings | None = None,
    store: LanceFileStore | None = None,
    embedder: OllamaEmbedder | None = None,
    reranker: FlashRankReranker | None = None,
) -> list[Candidate]:
    resolved = settings or get_settings()
    resolved_store = store or LanceFileStore(settings=resolved)
    query_vector = (embedder or OllamaEmbedder(settings=resolved)).embed(query)
    vector_results = resolved_store.vector_search(query_vector, resolved.retrieval.top_k)
    fts_results = resolved_store.fts_search(query, resolved.retrieval.top_k)
    candidates = fuse_rrf(
        vector_results,
        fts_results,
        rrf_k=resolved.retrieval.rrf_k,
        limit=resolved.retrieval.top_k,
    )
    return rerank_and_penalize(
        query,
        candidates,
        reranker or FlashRankReranker(settings=resolved),
        raw_status_penalty=resolved.retrieval.raw_status_penalty,
        limit=resolved.retrieval.retrieval_top_k,
    )


@click.group()
def app() -> None:
    """MindPalace Phase-1 CLI."""


@app.command("index")
def index_command() -> None:
    stats = index_vault()
    click.echo(f"Indexed {stats.chunks} chunks from {stats.documents} Markdown documents.")


@app.command("search")
@click.argument("query")
def search_command(query: str) -> None:
    for position, candidate in enumerate(search_chunks(query), start=1):
        chunk = candidate.payload
        click.echo(
            f"{position}. score={candidate.final_score:.6f} "
            f"status={candidate.status} source={chunk.doc_path}#{chunk.chunk_index}"
        )
        click.echo(candidate.text)
        click.echo()


if __name__ == "__main__":
    app()

