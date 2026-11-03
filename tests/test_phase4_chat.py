"""Chat REPL lifecycle: history, retrieval context, async extraction, shutdown."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any

import pytest
import requests

import palace.cli as cli
from palace.claims.extractor import ExtractedClaim, ExtractedClaims
from palace.domain.models import (
    Claim,
    ClaimProposal,
    ChunkStatus,
    DocumentChunk,
    Severity,
    SourceType,
)
from palace.retrieval.retrieval_types import Candidate


def _bob() -> ExtractedClaims:
    return ExtractedClaims(
        claims=[
            ExtractedClaim(
                entity="user",
                attribute="name",
                value="Bob",
                is_explicit=True,
                severity=Severity.LOW,
            )
        ]
    )


class _LLM:
    """Replaces cli.OllamaClient: streams tokens, serves fast-tier extraction."""

    instances: list["_LLM"] = []

    def __init__(self, **kwargs: Any) -> None:
        self.stream_calls: list[list[dict[str, str]]] = []
        self.stream_done = threading.Event()
        self.closed = False
        self.extraction: Callable[[], Any] = lambda: ExtractedClaims(claims=[])
        self.stream_errors: list[Exception | None] = []
        _LLM.instances.append(self)

    async def astream_messages(self, messages: Any, *, tier: str):
        assert tier == "reasoning"
        self.stream_calls.append([dict(m) for m in messages])
        error = self.stream_errors.pop(0) if self.stream_errors else None
        if error is not None:
            yield "partial "
            raise error
        yield "Hel"
        yield "lo"
        self.stream_done.set()

    def structured(self, prompt: str, response_model: Any, **kwargs: Any) -> Any:
        assert kwargs.get("tier") == "fast"
        result = self.extraction()
        if isinstance(result, Exception):
            raise result
        return result

    async def aclose(self) -> None:
        self.closed = True


class _Searcher:
    def __init__(self, candidates: list[Candidate] | None = None) -> None:
        self.candidates = candidates or []
        self.queries: list[str] = []

    async def retrieve(self, query: str) -> list[Candidate]:
        self.queries.append(query)
        return self.candidates


class _Embedder:
    def __init__(self, **kwargs: Any) -> None:
        _Embedder.calls = getattr(_Embedder, "calls", [])

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        _Embedder.calls.append(list(texts))
        return [[1.0] + [0.0] * 767 for _ in texts]


class _Store:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.proposals: list[ClaimProposal] = []

    def propose_claim(self, proposal: ClaimProposal, embedding: list[float]) -> Claim:
        self.proposals.append(proposal)
        return Claim(
            entity=proposal.entity,
            attribute=proposal.attribute,
            value=proposal.value,
            source_type=proposal.source_type,
        )


@pytest.fixture
def chat(monkeypatch: pytest.MonkeyPatch):
    """Wire the REPL to fakes; returns a handle for scripting and inspection."""

    class Handle:
        llm: _LLM
        searcher = _Searcher()
        store = _Store()
        extraction: Callable[[], Any] = staticmethod(lambda: ExtractedClaims(claims=[]))
        stream_errors: list[Exception | None] = []

        def script(self, *inputs: str) -> None:
            queue = iter(inputs)
            monkeypatch.setattr(
                cli.click, "prompt", lambda *a, **k: next(queue, "quit")
            )

    handle = Handle()
    _LLM.instances.clear()
    _Embedder.calls = []

    def make_llm(**kwargs: Any) -> _LLM:
        handle.llm = _LLM(**kwargs)
        handle.llm.extraction = lambda: handle.extraction()
        handle.llm.stream_errors = list(handle.stream_errors)
        return handle.llm

    monkeypatch.setattr(cli, "OllamaClient", make_llm)
    monkeypatch.setattr(cli, "OllamaEmbedder", _Embedder)
    monkeypatch.setattr(cli, "_ReopeningClaimsStoreAdapter", lambda settings: handle.store)
    monkeypatch.setattr(cli, "_build_searcher", lambda settings: handle.searcher)
    return handle


def _chunk_candidate(text: str) -> Candidate:
    chunk = DocumentChunk(
        doc_path="data/vault/network.md",
        chunk_index=2,
        text=text,
        vector=[0.0] * 768,
        status=ChunkStatus.COMPILED,
        source_type=SourceType.DOCUMENT,
    )
    return Candidate(
        id=chunk.chunk_id, text=text, kind="chunk", status="compiled", payload=chunk
    )


# ------------------------------------------------------------ conversation


async def test_multi_turn_history_reaches_the_model_in_order(chat, capsys) -> None:
    chat.script("my name is Bob", "what is my name?", "quit")

    await cli._chat_runner(dry_run=False)

    first, second = chat.llm.stream_calls
    assert [m["role"] for m in first] == ["system", "user"]
    assert [(m["role"], m["content"]) for m in second[1:]] == [
        ("user", "my name is Bob"),
        ("assistant", "Hello"),
        ("user", "what is my name?"),
    ]
    assert "Palace> Hello" in capsys.readouterr().out
    assert chat.llm.closed


async def test_retrieved_context_is_placed_in_the_system_prompt(chat) -> None:
    chat.searcher = _Searcher([_chunk_candidate("Pi-hole runs at 192.168.1.5")])
    chat.script("where is pihole?", "quit")

    await cli._chat_runner(dry_run=False)

    system = chat.llm.stream_calls[0][0]
    assert system["role"] == "system"
    assert "Pi-hole runs at 192.168.1.5" in system["content"]
    assert "data/vault/network.md chunk 2" in system["content"]
    assert chat.searcher.queries == ["where is pihole?"]


async def test_reply_failure_does_not_crash_or_poison_history(chat, capsys) -> None:
    chat.script("first question", "second question", "quit")
    chat.stream_errors = [requests.ConnectionError("ollama went away")]

    await cli._chat_runner(dry_run=False)

    first, second = chat.llm.stream_calls
    assert [m["content"] for m in second[1:]] == ["second question"]  # failed turn dropped
    assert "[error] could not answer" in capsys.readouterr().err


# ----------------------------------------------------- claim extraction flow


async def test_extraction_does_not_block_response_streaming(chat) -> None:
    saw_stream_done: list[bool] = []

    def extraction() -> ExtractedClaims:
        # Runs in a worker thread. It can only observe a finished stream if
        # the reply was generated while extraction was still in flight.
        saw_stream_done.append(chat.llm.stream_done.wait(timeout=3))
        return _bob()

    chat.script("my name is Bob", "quit")
    chat.extraction = extraction
    await cli._chat_runner(dry_run=False)

    assert saw_stream_done == [True]
    assert [p.value for p in chat.store.proposals] == ["Bob"]


async def test_shutdown_waits_for_pending_extraction_and_reports_it(chat, capsys) -> None:
    def slow() -> ExtractedClaims:
        time.sleep(0.3)
        return _bob()

    chat.script("my name is Bob", "quit")
    chat.extraction = slow
    await cli._chat_runner(dry_run=False)

    assert [p.value for p in chat.store.proposals] == ["Bob"]  # finished, not dropped
    assert "[claims] saved 1 claim(s): user:name=Bob (active)" in capsys.readouterr().out
    assert chat.llm.closed


@pytest.mark.parametrize(
    "failure, expected",
    [
        (ValueError("not json"), "malformed"),
        (requests.ConnectionError("refused"), "model or embedding service unavailable"),
    ],
)
async def test_extraction_failure_is_reported_but_chat_continues(
    chat, capsys, failure: Exception, expected: str
) -> None:
    chat.script("my name is Bob", "and I live in KL", "quit")
    chat.extraction = lambda: failure
    await cli._chat_runner(dry_run=False)

    captured = capsys.readouterr()
    assert len(chat.llm.stream_calls) == 2  # both turns still answered
    assert chat.store.proposals == []  # nothing written
    assert "[claims] claim extraction failed" in captured.err
    assert expected in captured.err


async def test_greetings_skip_extraction(chat) -> None:
    called: list[bool] = []
    chat.script("hello", "thanks!", "quit")
    chat.extraction = lambda: called.append(True) or ExtractedClaims(claims=[])
    await cli._chat_runner(dry_run=False)

    assert called == []


# ---------------------------------------------------------------- dry run


async def test_dry_run_shows_proposals_and_writes_nothing(chat, capsys) -> None:
    chat.script("my name is Bob", "quit")
    chat.extraction = _bob
    await cli._chat_runner(dry_run=True)

    out = capsys.readouterr().out
    assert "Proposed claims (dry run)" in out
    assert "Bob" in out and "name" in out
    assert chat.store.proposals == []
    assert _Embedder.calls == []


async def test_dry_run_surfaces_extraction_errors_instead_of_an_empty_table(
    chat, capsys
) -> None:
    chat.script("my name is Bob", "quit")
    chat.extraction = lambda: ValueError("garbled")
    await cli._chat_runner(dry_run=True)

    captured = capsys.readouterr()
    assert "No claims proposed." in captured.out
    assert "malformed" in captured.err and "garbled" in captured.err
    assert chat.store.proposals == []
