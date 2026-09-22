"""MindPalace command line interface."""

from __future__ import annotations

import click

from palace.config import Settings, get_settings
from palace.domain.models import (
    Claim,
    ClaimProposal,
    ClaimStatus,
    ChunkStatus,
    DocumentChunk,
    Severity,
    SourceType,
)
from palace.ingestion.indexer import index_vault
from palace.llm.embedder import OllamaEmbedder
from palace.retrieval.reranker import FlashRankReranker
from palace.retrieval.retrieval_types import Candidate
from palace.stores.file_store import LanceFileStore
from palace.stores.claims_store import SQLiteClaimsStore


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
    """MindPalace CLI."""


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


@app.group("claim")
def claim_group() -> None:
    """Manually add, list, and search structured claims."""


@claim_group.command("add")
@click.option("--entity", required=True, help="Entity the claim describes.")
@click.option("--attribute", required=True, help="Attribute asserted about the entity.")
@click.option("--value", required=True, help="Claim value to store and embed.")
@click.option("--inferred", is_flag=True, help="Treat the claim as inferred instead of explicit.")
@click.option(
    "--severity",
    type=click.Choice([severity.value for severity in Severity], case_sensitive=False),
    default=Severity.LOW.value,
    show_default=True,
)
def claim_add(entity: str, attribute: str, value: str, inferred: bool, severity: str) -> None:
    settings = get_settings()
    proposal = ClaimProposal(
        entity=entity,
        attribute=attribute,
        value=value,
        source_type=SourceType.CHAT,
        severity=Severity(severity.lower()),
        is_explicit=not inferred,
    )
    embedding = OllamaEmbedder(settings=settings).embed(value)
    with SQLiteClaimsStore(settings=settings) as store:
        claim = store.propose_claim(proposal, embedding)
    click.echo(
        f"Added claim {claim.id} status={claim.status.value} "
        f"key={claim.claim_key} value={claim.value}"
    )


def _print_claim_table(claims: list[Claim]) -> None:
    if not claims:
        click.echo("No claims found.")
        return
    headers = ("ID", "ENTITY", "ATTRIBUTE", "VALUE", "STATUS", "SEVERITY")
    rows = [
        (
            claim.id,
            claim.entity,
            claim.attribute,
            claim.value,
            claim.status.value,
            claim.severity.value,
        )
        for claim in claims
    ]
    widths = [max(len(header), *(len(row[index]) for row in rows)) for index, header in enumerate(headers)]
    template = "  ".join(f"{{:{width}}}" for width in widths)
    click.echo(template.format(*headers))
    click.echo(template.format(*("-" * width for width in widths)))
    for row in rows:
        click.echo(template.format(*row))


@claim_group.command("list")
@click.option(
    "--status",
    type=click.Choice([status.value for status in ClaimStatus], case_sensitive=False),
    default=None,
    help="Only show claims with this status.",
)
def claim_list(status: str | None) -> None:
    selected_status = ClaimStatus(status.lower()) if status else None
    with SQLiteClaimsStore() as store:
        claims = store.list_claims(selected_status)
    _print_claim_table(claims)


@claim_group.command("search")
@click.argument("query")
def claim_search(query: str) -> None:
    settings = get_settings()
    embedding = OllamaEmbedder(settings=settings).embed(query)
    with SQLiteClaimsStore(settings=settings) as store:
        claims = store.search_claims(
            embedding,
            query,
            settings.retrieval.claims_search_k,
        )
    _print_claim_table(claims)


if __name__ == "__main__":
    app()
