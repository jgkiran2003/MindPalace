"""MindPalace command line interface."""

from __future__ import annotations

import asyncio

import click

from palace.config import Settings, get_settings
from palace.domain.models import (
    Claim,
    ClaimProposal,
    ClaimStatus,
    ConflictGroup,
    Severity,
    SourceType,
)
from palace.ingestion.indexer import index_vault
from palace.llm.embedder import OllamaEmbedder
from palace.retrieval.hybrid_searcher import HybridSearcher
from palace.retrieval.reranker import FlashRankReranker
from palace.stores.claims_store import SQLiteClaimsStore
from palace.stores.file_store import LanceFileStore


class _ReopeningClaimsStoreAdapter:
    """Open SQLite in the calling thread for each protocol operation."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def get_active_claim(self, entity: str, attribute: str) -> Claim | None:
        with SQLiteClaimsStore(settings=self.settings) as store:
            return store.get_active_claim(entity, attribute)

    def search_claims(
        self,
        query_embedding: list[float],
        query_text: str,
        k: int,
        status_filter: list[ClaimStatus] | None = None,
    ) -> list[Claim]:
        with SQLiteClaimsStore(settings=self.settings) as store:
            return store.search_claims(
                query_embedding,
                query_text,
                k,
                status_filter,
            )

    def propose_claim(
        self,
        proposal: ClaimProposal,
        embedding: list[float],
    ) -> Claim:
        with SQLiteClaimsStore(settings=self.settings) as store:
            return store.propose_claim(proposal, embedding)

    def get_conflict_group(self, claim_key: str) -> ConflictGroup | None:
        with SQLiteClaimsStore(settings=self.settings) as store:
            return store.get_conflict_group(claim_key)


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
    settings = get_settings()
    searcher = HybridSearcher(
        file_store=LanceFileStore(settings=settings),
        claims_store=_ReopeningClaimsStoreAdapter(settings),
        embedder=OllamaEmbedder(settings=settings),
        reranker=FlashRankReranker(settings=settings),
        settings=settings,
    )
    results = asyncio.run(searcher.retrieve(query))

    for position, candidate in enumerate(results, start=1):
        if candidate.kind == "chunk":
            chunk = candidate.payload
            click.echo(
                f"{position}. [CHUNK] score={candidate.final_score:.6f} "
                f"status={candidate.status} "
                f"source={chunk.doc_path}#{chunk.chunk_index}"
            )
            click.echo(f"   {candidate.text}")
            click.echo()
            continue

        claim = candidate.payload
        click.echo(
            f"{position}. [CLAIM] score={candidate.final_score:.6f} "
            f"status={candidate.status} key={claim.claim_key} value={claim.value}"
        )
        if candidate.conflict_group is not None:
            click.echo("   CONFLICT GROUP:")
            for sibling in candidate.conflict_group.claims:
                click.echo(f"      - {sibling.value}")
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
    widths = [
        max(len(header), *(len(row[index]) for row in rows))
        for index, header in enumerate(headers)
    ]
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
