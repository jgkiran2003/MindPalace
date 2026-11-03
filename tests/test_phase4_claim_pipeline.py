"""Claim pipeline: success, empty, malformed, LLM/storage failure, dry-run."""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from typing import Any

import pytest
import requests

import palace.cli as cli
from palace.claims.extractor import (
    ExtractedClaim,
    ExtractedClaims,
    extract_claim_proposals,
)
from palace.claims.outcomes import (
    ClaimEmbeddingError,
    ClaimExtractionLLMError,
    ClaimOutcome,
    ClaimStorageError,
    MalformedExtractionError,
)
from palace.claims.resolver import resolve_and_store_proposals
from palace.config import Settings, get_settings
from palace.domain.models import ClaimProposal, ClaimStatus, Severity, SourceType
from palace.stores.claims_store import SQLiteClaimsStore

DIALOGUE = [{"role": "user", "content": "I work at MyStorage"}]


def _claim(value: str = "MyStorage", *, explicit: bool = True) -> ExtractedClaim:
    return ExtractedClaim(
        entity="user",
        attribute="employer",
        value=value,
        is_explicit=explicit,
        severity=Severity.LOW,
    )


class _LLM:
    """Fast-tier stub: returns ``result`` or raises ``error``."""

    def __init__(self, result: Any = None, error: Exception | None = None) -> None:
        self.result = result
        self.error = error
        self.calls = 0

    def structured(self, prompt: str, response_model: Any, **kwargs: Any) -> Any:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.result


class _Embedder:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[list[str]] = []

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        if self.error is not None:
            raise self.error
        return [[1.0] + [0.0] * 767 for _ in texts]


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    base = get_settings()
    return base.model_copy(
        update={
            "paths": base.paths.model_copy(
                update={"claims_db_path": str(tmp_path / "claims.db")}
            )
        }
    )


def _claim_count(settings: Settings) -> int:
    with SQLiteClaimsStore(settings=settings) as store:
        return store.db.execute("SELECT count(*) FROM claims").fetchone()[0]


async def _run(
    llm: Any,
    embedder: Any,
    store: Any,
    *,
    dry_run: bool = False,
) -> cli.ClaimExchangeResult:
    return await cli._process_claim_exchange(
        DIALOGUE,
        dry_run=dry_run,
        llm=llm,
        embedder=embedder,
        claims_store=store,
        lock=asyncio.Lock(),
    )


# --------------------------------------------------------------- extractor


@pytest.mark.parametrize(
    "error",
    [
        requests.ConnectionError("refused"),
        requests.Timeout("slow"),
        requests.HTTPError("404 model not found"),
    ],
)
def test_extractor_classifies_service_failures(error: Exception) -> None:
    with pytest.raises(ClaimExtractionLLMError):
        extract_claim_proposals("User: hi", client=_LLM(error=error))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "error",
    [
        ValueError("Ollama response did not contain structured content"),
        requests.exceptions.JSONDecodeError("not json", "doc", 0),
    ],
)
def test_extractor_classifies_malformed_output(error: Exception) -> None:
    with pytest.raises(MalformedExtractionError):
        extract_claim_proposals("User: hi", client=_LLM(error=error))  # type: ignore[arg-type]


def test_extractor_classifies_schema_violation_as_malformed() -> None:
    class _Invalid:
        def structured(self, prompt: str, response_model: Any, **kwargs: Any) -> Any:
            return response_model.model_validate_json('{"claims":[{"entity":"user"}]}')

    with pytest.raises(MalformedExtractionError):
        extract_claim_proposals("User: hi", client=_Invalid())  # type: ignore[arg-type]


# ---------------------------------------------------------------- resolver


def test_resolver_embedding_failure_writes_nothing(settings: Settings) -> None:
    proposal = ClaimProposal(
        entity="user", attribute="a", value="v", source_type=SourceType.CHAT
    )
    store = cli._ReopeningClaimsStoreAdapter(settings)
    with pytest.raises(ClaimEmbeddingError):
        resolve_and_store_proposals(
            [proposal], _Embedder(error=ConnectionError("down")), store  # type: ignore[arg-type]
        )
    assert _claim_count(settings) == 0


def test_resolver_embedding_count_mismatch_is_an_embedding_error() -> None:
    class _Short:
        def embed_batch(self, texts: list[str]) -> list[list[float]]:
            return []

    proposal = ClaimProposal(
        entity="user", attribute="a", value="v", source_type=SourceType.CHAT
    )
    with pytest.raises(ClaimEmbeddingError):
        resolve_and_store_proposals([proposal], _Short(), object())  # type: ignore[arg-type]


# ---------------------------------------------------------------- pipeline


async def test_successful_extraction_is_persisted(settings: Settings) -> None:
    store = cli._ReopeningClaimsStoreAdapter(settings)
    llm = _LLM(ExtractedClaims(claims=[_claim()]))

    result = await _run(llm, _Embedder(), store)

    assert result.outcome is ClaimOutcome.SUCCESS
    assert result.ok and not result.dry_run
    assert [(c.claim_key, c.value) for c in result.claims] == [
        ("user:employer", "MyStorage")
    ]
    assert store.get_active_claim("user", "employer").value == "MyStorage"  # type: ignore[union-attr]
    assert "user:employer=MyStorage (active)" in (result.describe() or "")


async def test_explicit_restatement_supersedes_through_the_pipeline(
    settings: Settings,
) -> None:
    store = cli._ReopeningClaimsStoreAdapter(settings)
    await _run(_LLM(ExtractedClaims(claims=[_claim("OldCo")])), _Embedder(), store)
    await _run(_LLM(ExtractedClaims(claims=[_claim("NewCo")])), _Embedder(), store)

    with SQLiteClaimsStore(settings=settings) as reader:
        by_value = {c.value: c.status for c in reader.list_claims()}
    assert by_value == {"OldCo": ClaimStatus.SUPERSEDED, "NewCo": ClaimStatus.ACTIVE}


async def test_inferred_conflict_is_flagged_for_review(settings: Settings) -> None:
    store = cli._ReopeningClaimsStoreAdapter(settings)
    await _run(_LLM(ExtractedClaims(claims=[_claim("A")])), _Embedder(), store)
    await _run(
        _LLM(ExtractedClaims(claims=[_claim("B", explicit=False)])), _Embedder(), store
    )

    with SQLiteClaimsStore(settings=settings) as reader:
        assert {c.status for c in reader.list_claims()} == {ClaimStatus.NEEDS_REVIEW}
        assert reader.get_conflict_group("user:employer") is not None


async def test_empty_extraction_is_no_claims_and_writes_nothing(
    settings: Settings,
) -> None:
    embedder = _Embedder()
    result = await _run(
        _LLM(ExtractedClaims(claims=[])), embedder, cli._ReopeningClaimsStoreAdapter(settings)
    )

    assert result.outcome is ClaimOutcome.NO_CLAIMS
    assert result.ok and result.describe() is None
    assert embedder.calls == []
    assert _claim_count(settings) == 0


async def test_malformed_output_is_reported_and_writes_nothing(
    settings: Settings,
) -> None:
    embedder = _Embedder()
    result = await _run(
        _LLM(error=ValueError("bad json")), embedder, cli._ReopeningClaimsStoreAdapter(settings)
    )

    assert result.outcome is ClaimOutcome.MALFORMED_OUTPUT
    assert not result.ok
    assert "bad json" in (result.error or "")
    assert "malformed" in (result.describe() or "")
    assert embedder.calls == []
    assert _claim_count(settings) == 0


@pytest.mark.parametrize("error", [requests.ConnectionError("x"), requests.Timeout("x")])
async def test_llm_failure_is_reported_and_writes_nothing(
    settings: Settings, error: Exception
) -> None:
    result = await _run(
        _LLM(error=error), _Embedder(), cli._ReopeningClaimsStoreAdapter(settings)
    )

    assert result.outcome is ClaimOutcome.LLM_FAILURE
    assert _claim_count(settings) == 0


async def test_embedding_failure_is_reported_and_writes_nothing(
    settings: Settings,
) -> None:
    result = await _run(
        _LLM(ExtractedClaims(claims=[_claim()])),
        _Embedder(error=requests.ConnectionError("embed down")),
        cli._ReopeningClaimsStoreAdapter(settings),
    )

    assert result.outcome is ClaimOutcome.LLM_FAILURE
    assert [p.value for p in result.proposals] == ["MyStorage"]
    assert _claim_count(settings) == 0


class _FailingStore:
    """Delegates to a real store but fails on the ``fail_on``-th propose."""

    def __init__(self, inner: Any, fail_on: int) -> None:
        self.inner = inner
        self.fail_on = fail_on
        self.calls = 0

    def propose_claim(self, proposal: ClaimProposal, embedding: list[float]) -> Any:
        self.calls += 1
        if self.calls == self.fail_on:
            raise sqlite3.OperationalError("database is locked")
        return self.inner.propose_claim(proposal, embedding)


async def test_storage_failure_is_reported(settings: Settings) -> None:
    store = _FailingStore(cli._ReopeningClaimsStoreAdapter(settings), fail_on=1)

    result = await _run(_LLM(ExtractedClaims(claims=[_claim()])), _Embedder(), store)

    assert result.outcome is ClaimOutcome.STORAGE_FAILURE
    assert "database is locked" in (result.error or "")
    assert result.claims == []
    assert _claim_count(settings) == 0


async def test_partial_storage_failure_reports_what_was_saved(
    settings: Settings,
) -> None:
    store = _FailingStore(cli._ReopeningClaimsStoreAdapter(settings), fail_on=2)
    llm = _LLM(
        ExtractedClaims(
            claims=[
                _claim("MyStorage"),
                ExtractedClaim(
                    entity="user",
                    attribute="city",
                    value="Kuala Lumpur",
                    is_explicit=True,
                    severity=Severity.LOW,
                ),
            ]
        )
    )

    result = await _run(llm, _Embedder(), store)

    assert result.outcome is ClaimOutcome.STORAGE_FAILURE
    assert [c.value for c in result.claims] == ["MyStorage"]
    assert _claim_count(settings) == 1
    assert "1 claim(s) were saved" in (result.describe() or "")


async def test_unexpected_extractor_error_is_not_swallowed(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("surprise")

    monkeypatch.setattr(cli, "extract_claim_proposals", boom)
    result = await _run(
        _LLM(), _Embedder(), cli._ReopeningClaimsStoreAdapter(settings)
    )

    assert result.outcome is ClaimOutcome.UNEXPECTED_ERROR
    assert "surprise" in (result.error or "")


async def test_dry_run_exposes_proposals_without_touching_state(
    settings: Settings,
) -> None:
    embedder = _Embedder()
    store = cli._ReopeningClaimsStoreAdapter(settings)

    result = await _run(
        _LLM(ExtractedClaims(claims=[_claim()])), embedder, store, dry_run=True
    )

    assert result.outcome is ClaimOutcome.SUCCESS and result.dry_run
    assert [p.value for p in result.proposals] == ["MyStorage"]
    assert embedder.calls == []
    assert not Path(settings.paths.claims_db_path).exists()
    assert "dry run" in (result.describe() or "")


async def test_dry_run_leaves_an_existing_database_unchanged(
    settings: Settings,
) -> None:
    store = cli._ReopeningClaimsStoreAdapter(settings)
    await _run(_LLM(ExtractedClaims(claims=[_claim("Before")])), _Embedder(), store)
    with SQLiteClaimsStore(settings=settings) as reader:
        before = [(c.id, c.value, c.status, c.updated_at) for c in reader.list_claims()]

    await _run(
        _LLM(ExtractedClaims(claims=[_claim("After")])), _Embedder(), store, dry_run=True
    )

    with SQLiteClaimsStore(settings=settings) as reader:
        after = [(c.id, c.value, c.status, c.updated_at) for c in reader.list_claims()]
    assert after == before


def test_storage_error_carries_persisted_claims() -> None:
    error = ClaimStorageError("x")
    assert error.persisted == []
