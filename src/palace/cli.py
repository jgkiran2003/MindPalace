"""MindPalace command line interface."""

from __future__ import annotations

import asyncio
from pathlib import Path

import click

from palace.claims.extractor import extract_claim_proposals
from palace.claims.outcomes import (
    ClaimEmbeddingError,
    ClaimExchangeResult,
    ClaimExtractionLLMError,
    ClaimOutcome,
    ClaimStorageError,
    MalformedExtractionError,
)
from palace.claims.resolver import resolve_and_store_proposals
from palace.compilation.compiler import CompilationError, compile_raw_notes
from palace.config import Settings, get_settings
from palace.domain.models import (
    Claim,
    ClaimProposal,
    ClaimStatus,
    ConflictGroup,
    Severity,
    SourceType,
)
from palace.ingestion.indexer import VaultIndexer, index_vault
from palace.llm.client import OllamaClient
from palace.llm.embedder import OllamaEmbedder
from palace.retrieval.hybrid_searcher import HybridSearcher
from palace.retrieval.reranker import FlashRankReranker
from palace.retrieval.retrieval_types import Candidate
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


def _build_searcher(settings: Settings) -> HybridSearcher:
    return HybridSearcher(
        file_store=LanceFileStore(settings=settings),
        claims_store=_ReopeningClaimsStoreAdapter(settings),
        embedder=OllamaEmbedder(settings=settings),
        reranker=FlashRankReranker(settings=settings),
        settings=settings,
    )


@click.group()
def app() -> None:
    """MindPalace CLI."""


@app.command("index")
def index_command() -> None:
    stats = index_vault()
    message = f"Indexed {stats.chunks} chunks from {stats.documents} Markdown documents."
    if stats.archived:
        message += f" Skipped {stats.archived} archived (already compiled) raw notes."
    click.echo(message)


@app.command("search")
@click.argument("query")
def search_command(query: str) -> None:
    results = asyncio.run(_build_searcher(get_settings()).retrieve(query))
    _print_candidates(results)


def _print_candidates(results: list[Candidate]) -> None:
    if not results:
        click.echo("No results found.")
        return
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


def _context_for_prompt(candidates: list[Candidate]) -> str:
    if not candidates:
        return "No relevant stored context was found."
    sections: list[str] = []
    for candidate in candidates:
        if candidate.kind == "chunk":
            chunk = candidate.payload
            label = f"document {chunk.doc_path} chunk {chunk.chunk_index}"
        else:
            claim = candidate.payload
            label = f"claim {claim.claim_key} ({claim.status.value})"
        sections.append(f"[{label}]\n{candidate.text}")
    return "\n\n".join(sections)


def _print_proposal_table(proposals: list[ClaimProposal]) -> None:
    if not proposals:
        click.echo("No claims proposed.")
        return
    headers = ("Entity", "Attribute", "Value", "Is Explicit", "Severity")
    rows = [
        (
            proposal.entity,
            proposal.attribute,
            proposal.value,
            "yes" if proposal.is_explicit else "no",
            proposal.severity.value,
        )
        for proposal in proposals
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


TRIVIAL_CHAT = {"hi", "hello", "hey", "thanks", "ok", "okay"}


async def _process_claim_exchange(
    dialogue: list[dict[str, str]],
    *,
    dry_run: bool,
    llm: OllamaClient,
    embedder: OllamaEmbedder,
    claims_store: _ReopeningClaimsStoreAdapter,
    lock: asyncio.Lock,
) -> ClaimExchangeResult:
    """Extract and resolve user claims, reporting exactly what happened.

    Nothing is written unless extraction produced valid proposals; every
    failure is returned as a distinct outcome rather than an empty list.
    """
    exchange_text = "\n".join(
        f"{message['role'].capitalize()}: {message['content']}"
        for message in dialogue
    )
    async with lock:
        try:
            proposals = await asyncio.to_thread(
                extract_claim_proposals,
                exchange_text,
                client=llm,
            )
        except MalformedExtractionError as exc:
            return ClaimExchangeResult(
                ClaimOutcome.MALFORMED_OUTPUT, dry_run=dry_run, error=str(exc)
            )
        except ClaimExtractionLLMError as exc:
            return ClaimExchangeResult(
                ClaimOutcome.LLM_FAILURE, dry_run=dry_run, error=str(exc)
            )
        except Exception as exc:
            return ClaimExchangeResult(
                ClaimOutcome.UNEXPECTED_ERROR,
                dry_run=dry_run,
                error=f"{type(exc).__name__}: {exc}",
            )
        if not proposals:
            return ClaimExchangeResult(ClaimOutcome.NO_CLAIMS, dry_run=dry_run)

        try:
            claims = await asyncio.to_thread(
                resolve_and_store_proposals,
                proposals,
                embedder,
                claims_store,
                dry_run=dry_run,
            )
        except ClaimEmbeddingError as exc:
            return ClaimExchangeResult(
                ClaimOutcome.LLM_FAILURE,
                dry_run=dry_run,
                proposals=proposals,
                error=str(exc),
            )
        except ClaimStorageError as exc:
            return ClaimExchangeResult(
                ClaimOutcome.STORAGE_FAILURE,
                dry_run=dry_run,
                proposals=proposals,
                claims=exc.persisted,
                error=str(exc),
            )
        except Exception as exc:
            return ClaimExchangeResult(
                ClaimOutcome.UNEXPECTED_ERROR,
                dry_run=dry_run,
                proposals=proposals,
                error=f"{type(exc).__name__}: {exc}",
            )
        return ClaimExchangeResult(
            ClaimOutcome.SUCCESS, dry_run=dry_run, proposals=proposals, claims=claims
        )


def _report_claim_result(result: ClaimExchangeResult) -> None:
    summary = result.describe()
    if summary is not None:
        click.echo(f"[claims] {summary}", err=not result.ok)


def _report_finished_claim_tasks(
    tasks: list[asyncio.Task[ClaimExchangeResult]],
) -> None:
    """Print (once) the outcome of background claim tasks that have finished."""
    for task in list(tasks):
        if not task.done():
            continue
        tasks.remove(task)
        if task.cancelled():
            click.echo("[claims] claim extraction was cancelled", err=True)
        elif task.exception() is not None:
            click.echo(f"[claims] unexpected error: {task.exception()}", err=True)
        else:
            _report_claim_result(task.result())


async def _stream_reply(
    searcher: HybridSearcher,
    llm: OllamaClient,
    history: list[dict[str, str]],
    user_input: str,
) -> str:
    candidates = await searcher.retrieve(user_input)
    context = _context_for_prompt(candidates)
    messages = [
        {
            "role": "system",
            "content": (
                "Use the supplied context when relevant, distinguish stored "
                "facts from inference, and do not invent personal details.\n\n"
                f"Retrieved context:\n{context}"
            ),
        },
        *history,
        {"role": "user", "content": user_input},
    ]
    print("Palace> ", end="", flush=True)
    reply_parts: list[str] = []
    async for token in llm.astream_messages(messages, tier="reasoning"):
        reply_parts.append(token)
        print(token, end="", flush=True)
    print()
    return "".join(reply_parts)


@app.command("chat")
@click.option(
    "--dry-run",
    is_flag=True,
    help="Preview extracted claims without saving them.",
)
def chat_command(dry_run: bool) -> None:
    """Start an interactive, retrieval-grounded conversation."""
    asyncio.run(_chat_runner(dry_run))


async def _chat_runner(dry_run: bool) -> None:
    # Conversation history lives in memory for this session only; Phase 4 has
    # no persistent chat store (durable knowledge goes through claims).
    settings = get_settings()
    llm = OllamaClient(settings=settings)
    embedder = OllamaEmbedder(settings=settings)
    claims_store = _ReopeningClaimsStoreAdapter(settings)
    history: list[dict[str, str]] = []
    searcher: HybridSearcher | None = None
    claim_lock = asyncio.Lock()
    unreported: list[asyncio.Task[ClaimExchangeResult]] = []

    click.echo("MindPalace chat. Type 'quit' or 'exit' to leave.")
    try:
        while True:
            _report_finished_claim_tasks(unreported)
            try:
                user_input = (
                    await asyncio.to_thread(click.prompt, "You", prompt_suffix="> ")
                ).strip()
            except (click.Abort, EOFError, KeyboardInterrupt):
                click.echo("\nGoodbye.")
                break
            if not user_input:
                continue
            if user_input.casefold() in {"quit", "exit"}:
                click.echo("Goodbye.")
                break

            cleaned = user_input.casefold().strip(" \t\r\n.!?,")
            claim_task: asyncio.Task[ClaimExchangeResult] | None = None
            if cleaned not in TRIVIAL_CHAT:
                dialogue = [
                    dict(message)
                    for message in history[-2:]
                ]
                dialogue.append({"role": "user", "content": user_input})
                claim_task = asyncio.create_task(
                    _process_claim_exchange(
                        dialogue,
                        dry_run=dry_run,
                        llm=llm,
                        embedder=embedder,
                        claims_store=claims_store,
                        lock=claim_lock,
                    )
                )
                if not dry_run:
                    unreported.append(claim_task)

            try:
                if searcher is None:
                    searcher = await asyncio.to_thread(_build_searcher, settings)
                reply = await _stream_reply(searcher, llm, history, user_input)
            except Exception as exc:
                click.echo(f"\n[error] could not answer: {type(exc).__name__}: {exc}", err=True)
            else:
                history.extend(
                    [
                        {"role": "user", "content": user_input},
                        {"role": "assistant", "content": reply},
                    ]
                )

            if dry_run and claim_task is not None:
                result = await claim_task
                click.echo("─" * 72)
                click.echo("Proposed claims (dry run)")
                _print_proposal_table(result.proposals)
                if not result.ok:
                    _report_claim_result(result)
                click.echo("─" * 72)
    finally:
        # Let in-flight extractions finish (and report) before closing.
        if unreported:
            await asyncio.gather(*unreported, return_exceptions=True)
            _report_finished_claim_tasks(unreported)
        await llm.aclose()


@app.command("compile")
def compile_command() -> None:
    """Compile raw notes into one indexed note and archive the raw originals."""
    settings = get_settings()
    try:
        output = compile_raw_notes(
            Path(settings.paths.vault_dir),
            client=OllamaClient(settings=settings),
            indexer=VaultIndexer(settings=settings),
        )
    except CompilationError as exc:
        raise click.ClickException(str(exc)) from exc
    except OSError as exc:
        raise click.ClickException(
            f"compilation failed, nothing was archived: {type(exc).__name__}: {exc}"
        ) from exc
    if output is None:
        click.echo("Not enough unprocessed raw notes to compile.")
    else:
        click.echo(f"Created and indexed compiled note: {output}")
        click.echo(
            "Source notes were marked processed and removed from the search index; "
            "the files remain in the vault."
        )


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
