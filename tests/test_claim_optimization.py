from __future__ import annotations

import asyncio
import threading
from typing import Any

import pytest

import palace.cli as cli
from palace.claims.resolver import resolve_and_store_proposals
from palace.config import get_settings
from palace.domain.models import Claim, ClaimProposal, SourceType
from palace.llm.embedder import OllamaEmbedder


def _proposal(attribute: str, value: str) -> ClaimProposal:
    return ClaimProposal(
        entity="user",
        attribute=attribute,
        value=value,
        source_type=SourceType.CHAT,
        is_explicit=True,
    )


class _BatchEmbedder:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [[float(index)] * 768 for index, _ in enumerate(texts)]


class _ClaimsStore:
    def __init__(self) -> None:
        self.calls: list[tuple[ClaimProposal, list[float]]] = []

    def propose_claim(
        self,
        proposal: ClaimProposal,
        embedding: list[float],
    ) -> Claim:
        self.calls.append((proposal, embedding))
        return Claim(
            entity=proposal.entity,
            attribute=proposal.attribute,
            value=proposal.value,
            source_type=proposal.source_type,
        )


def test_resolver_embeds_all_proposals_in_one_batch() -> None:
    proposals = [
        _proposal("employer", "MyStorage"),
        _proposal("city", "Kuala Lumpur"),
        _proposal("language", "Python"),
    ]
    embedder = _BatchEmbedder()
    store = _ClaimsStore()

    resolved = resolve_and_store_proposals(
        proposals,
        embedder,
        store,  # type: ignore[arg-type]
    )

    assert embedder.calls == [["MyStorage", "Kuala Lumpur", "Python"]]
    assert len(store.calls) == 3
    assert [claim.value for claim in resolved] == [
        "MyStorage",
        "Kuala Lumpur",
        "Python",
    ]


class _EmbeddingResponse:
    def __init__(self, embeddings: list[list[float]]) -> None:
        self.embeddings = embeddings

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict[str, Any]:
        return {"embeddings": self.embeddings}


class _EmbeddingSession:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def post(self, url: str, **kwargs: Any) -> _EmbeddingResponse:
        self.calls.append({"url": url, **kwargs})
        inputs = kwargs["json"]["input"]
        return _EmbeddingResponse([[0.0] * 768 for _ in inputs])


def test_ollama_embedder_uses_one_native_batch_request() -> None:
    session = _EmbeddingSession()
    embedder = OllamaEmbedder(
        settings=get_settings(),
        session=session,  # type: ignore[arg-type]
    )

    vectors = embedder.embed_batch(["one", "two", "three"])

    assert len(vectors) == 3
    assert len(session.calls) == 1
    assert session.calls[0]["url"].endswith("/api/embed")
    assert session.calls[0]["json"]["input"] == ["one", "two", "three"]


class _ChatLLM:
    closed = False

    def __init__(self, **kwargs: Any) -> None:
        pass

    async def astream_messages(self, messages: Any, *, tier: str):
        yield "answer"

    async def aclose(self) -> None:
        self.closed = True


class _Searcher:
    def __init__(self, extraction_started: threading.Event | None = None) -> None:
        self.extraction_started = extraction_started

    async def retrieve(self, query: str) -> list[Any]:
        if self.extraction_started is not None:
            assert self.extraction_started.wait(timeout=2)
        return []


@pytest.mark.asyncio
async def test_chat_reprompts_while_claim_extraction_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extraction_started = threading.Event()
    release_extraction = threading.Event()
    extraction_finished = threading.Event()
    prompts = 0
    extracted_dialogue: list[str] = []

    def prompt(*args: Any, **kwargs: Any) -> str:
        nonlocal prompts
        prompts += 1
        if prompts == 1:
            return "my name is Bob"
        assert extraction_started.wait(timeout=2)
        release_extraction.set()
        return "quit"

    def extract(text: str, **kwargs: Any) -> list[ClaimProposal]:
        extracted_dialogue.append(text)
        extraction_started.set()
        assert release_extraction.wait(timeout=2)
        extraction_finished.set()
        return []

    monkeypatch.setattr(cli.click, "prompt", prompt)
    monkeypatch.setattr(cli, "OllamaClient", _ChatLLM)
    monkeypatch.setattr(
        cli, "_build_searcher", lambda settings: _Searcher(extraction_started)
    )
    monkeypatch.setattr(cli, "extract_claim_proposals", extract)

    await cli._chat_runner(dry_run=False)

    assert prompts == 2
    assert extraction_started.is_set()
    assert extraction_finished.is_set()
    assert extracted_dialogue == ["User: my name is Bob"]


@pytest.mark.asyncio
async def test_chat_skips_claim_extraction_for_trivial_greeting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replies = iter(["hello", "quit"])
    extraction_calls = 0

    def extract(*args: Any, **kwargs: Any) -> list[ClaimProposal]:
        nonlocal extraction_calls
        extraction_calls += 1
        return []

    monkeypatch.setattr(cli.click, "prompt", lambda *args, **kwargs: next(replies))
    monkeypatch.setattr(cli, "OllamaClient", _ChatLLM)
    monkeypatch.setattr(cli, "_build_searcher", lambda settings: _Searcher())
    monkeypatch.setattr(cli, "extract_claim_proposals", extract)

    await cli._chat_runner(dry_run=False)

    assert extraction_calls == 0
